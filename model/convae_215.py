import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.utils as utils
# from utils import utils
import time
from util.enthalpy_predictor import predict_enthalpy


class PriorBlock(nn.Module):
    def __init__(self, cond_dim, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        self.mu = torch.nn.Linear(64, latent_dim)
        self.logvar = torch.nn.Linear(64, latent_dim)

    def forward(self, c):
        h = self.mlp(c)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

class CondEncoder(nn.Module): # 通过embedding压缩smiles维度，使用rnn实现encoder，decoder采用mlp。未实现多层条件拼接
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim, state_fname, device, embed_dim=32) -> None:
        super().__init__()
        self.state_fname = state_fname
        self.device = device
        self.maxLength = maxLength
        self.num_vocabs = num_vocabs
        self.embed = nn.Embedding(num_vocabs, embed_dim, device=self.device)
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(embed_dim, fc_dims[0], device=self.device),
            nn.Linear(embed_dim * maxLength, fc_dims[0], device=self.device),  # deepseek改的
            # nn.BatchNorm1d(fc_dims[0]),  # 标准化,bn层放在激活函数前后都各有说法
            nn.ReLU()
        )
        for i in range(1, len(fc_dims)):  # [1,3),试过了不会输出i=3的情况
            self.fc.append(torch.nn.Linear(
                fc_dims[i - 1], fc_dims[i], device=self.device))
            # self.fc.append(nn.BatchNorm1d(fc_dims[i]))  # 标准化,bn层放在激活函数前后都各有说法
            self.fc.append(torch.nn.ReLU())
        self.prior_block = PriorBlock(cond_dim=con_dims, latent_dim=latent_dim).to(device)
        self.mu = torch.nn.Linear(fc_dims[-1], latent_dim, device=self.device)
        self.logvar = torch.nn.Linear(
            fc_dims[-1], latent_dim, device=self.device)  #mu和logvar的拟合依靠kld散度损失来更新

    def forward(self, X, enthalpy):
        X_embed = self.embed(X)
        X_flatten = torch.flatten(X_embed, start_dim=1)
        X_fc = self.fc(X_flatten)
        mu_x = self.mu(X_fc)
        logvar_x = self.logvar(X_fc)

        mu_prior, logvar_prior = self.prior_block(enthalpy.unsqueeze(1))
        mu = 0.6*mu_x + 0.4*mu_prior
        logvar = 0.6*logvar_x + 0.4*logvar_prior
        # enthalpy_expanded = enthalpy.unsqueeze(1).unsqueeze(2)  # (512, 1, 1)
        # enthalpy_expanded = enthalpy_expanded.expand(-1, X.size(1), -1)  # (512, 128,1)
        # X = torch.cat((X.to(self.device), enthalpy_expanded.to(self.device)), dim=2)  # (512, 128,18)
        # X = self.fc(X)  # (512, 128)
        # mu, logvar = self.mu(X), self.logvar(X)  # (512, 64)
        return self.reparameterize(mu, logvar), mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def loadState(self):
        if os.path.isfile(self.state_fname):
            self.load_state_dict(torch.load(self.state_fname))
        else:
            print("state file is not found")

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)


class CondDecoder(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, con_dims, latent_dim, hidden_dim, num_hidden, state_fname,
                 device, embed_dim=32) -> None:
        super().__init__()
        self.state_fname = state_fname
        self.device = device
        self.maxLength = maxLength
        # num_vocabs 是去除了 <start> 和 <end> 后的大小
        self.num_vocabs = num_vocabs # + 2  # 为了包含 <start> 和 <end>
        self.con_dims = con_dims
        # 修改 Embedding 层的词表大小
        self.embed = torch.nn.Embedding(num_vocabs, embed_dim, device=self.device)  
        
        # GRU 的输入维度应该是 latent_dim + embed_dim + con_dims
        input_dim = latent_dim + embed_dim + con_dims
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_hidden,
            batch_first=True,
            device=self.device
        )
        self.fc = torch.nn.Linear(hidden_dim, num_vocabs, device=self.device)

    def forward(self, latent_vec, enthalpy, inp, freerun=False, randomchoose=True, condition=True):
        latent_vec = latent_vec.to(self.device)  # (512, 64)
        enthalpy_ori = enthalpy
        enthalpy = enthalpy.unsqueeze(1).unsqueeze(2).expand(-1, self.maxLength, -1)  # (512, 128, 1)

        if not freerun:
            inp_embed = self.embed(inp)  # [batch_size, maxLength, embed_dim]
            batch_size = inp_embed.size(0)
            # 检查索引是否超出范围 (使用完整词表大小)
            if torch.max(inp) >= self.num_vocabs:
                print(torch.max(inp))
                raise ValueError(f"Input indices exceed vocabulary size: max index = {torch.max(inp)}, vocab size = {self.num_vocabs}")
            # 构建初始输入（右移一位）
            start_embed = self.embed(  # <start> 索引为0
                torch.zeros((batch_size, 1), dtype=torch.long, device=self.device))  # [batch, 1, embed_dim]
            # 右移一位
            inp_shifted = torch.cat([start_embed, inp_embed[:, :-1, :]], dim=1)  # [batch, maxLength, embed_dim]
            # 拼接潜在变量、嵌入输入和条件
            latent_expanded = latent_vec.unsqueeze(1).expand(-1, self.maxLength, -1)  # [batch, maxLength, latent_dim]
            X = torch.cat([latent_expanded, inp_shifted, enthalpy], dim=-1)  # [batch, maxLength, latent_dim + embed_dim + con_dims]

            # GRU 前向
            gru_out, _ = self.gru(X)
            logits = self.fc(gru_out)  # [batch, maxLength, num_vocabs]
            # 将概率分布转换为整数索引
            indices = torch.argmax(logits, dim=-1)  # [batch, maxLength]
            # 确保索引在合法范围内
            indices = torch.clamp(indices, 0, self.num_vocabs - 1)
            return logits  # indices.to(self.device)
        else:  # 生成模式：自由运行
            batch_size = latent_vec.size(0)
            out = torch.zeros((batch_size, self.maxLength), dtype=torch.float32, device=self.device)  # 添加 device
            cond = enthalpy_ori.unsqueeze(1).unsqueeze(2)  # [batch, 1, con_dims]
            # 初始字符设为 <start>（假设索引为0）
            current_idx = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
            current_embed = self.embed(current_idx)  # [batch, 1, embed_dim]

            # 初始化 GRU 隐藏状态
            hidden = None

            # 逐字符生成
            for i in range(self.maxLength):
                # 拼接当前输入
                X = torch.cat([
                    latent_vec.unsqueeze(1),  # [batch, 1, latent_dim]
                    current_embed,  # [batch, 1, embed_dim]
                    cond  # [batch, 1, con_dims]
                ], dim=-1)

                # GRU 前向传播
                gru_out, hidden = self.gru(X, hidden)  # 使用前一个隐藏状态
                logits = self.fc(gru_out)  # [batch, 1, num_vocabs]

                # 获取下一个字符
                if randomchoose:
                    probs = torch.softmax(logits.squeeze(1), dim=-1)
                    next_idx = torch.multinomial(probs, 1)  # [batch, 1]
                else:
                    next_idx = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # [batch, 1]

                # 更新输出和下一个输入
                out[:, i] = next_idx.squeeze(1)
                current_embed = self.embed(next_idx)  # [batch, 1, embed_dim]

            return out

    def loadState(self):
        if os.path.isfile(self.state_fname):
            self.load_state_dict(torch.load(self.state_fname))
        else:
            print("state file is not found")

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)


class ConVAE(object):
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim, hidden_dim, num_hidden,
                 encoder_state_fname, decoder_state_fname, device) -> None:
        self.num_vocabs = num_vocabs
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = CondEncoder(maxLength, num_vocabs, con_dims,
                               fc_dims, latent_dim, encoder_state_fname, device)
        self.decoder = CondDecoder(maxLength, num_vocabs, con_dims, latent_dim,
                               hidden_dim, num_hidden, decoder_state_fname, device)

    def reconstruction_quality_per_sample(self, X, enthalpy):
        """计算重构质量（基于整数索引输入）
        Args:
            X: [batch_size, maxLength] 输入的整数索引分子表示
            enthalpy: [batch_size] 生成焓条件
        Returns:
            correct_counts: [batch_size] 每个样本正确预测的字符数量
        """
        self.encoder.eval()
        self.decoder.eval()
        # 1. 通过编码器获取隐空间表示
        latent_vec, mu, logvar = self.encoder(X, enthalpy)
        # 2. 通过解码器重构输入（假设X已经是整数索引）
        pred_logits = self.decoder(mu, enthalpy, X)  # [batch, maxLength, num_vocabs]
        # 3. 计算每个位置的最大概率索引
        pred_indices = torch.argmax(pred_logits, dim=-1)  # [batch, maxLength]
        # 4. 计算正确字符数量
        correct_mask = (pred_indices == X)  # [batch, maxLength]
        correct_counts = correct_mask.sum(dim=-1)  # [batch]

        return correct_counts.float()  # 返回浮点数张量

    def sample(self, nSample):
        """从隐空间采样
        Args:
            nSample: 采样数量
        Returns:
            生成的token序列
        """
        # 从标准正态分布采样
        latent_vec = torch.randn( # 这里采样生成的是[nSample, maxlength]
            (nSample, self.latent_dim), device=self.device)
        _enthalpy = torch.randn(nSample, device=self.device)  # 这里随机采样的生成焓也随机生成
        # 通过解码器生成分子
        y = self.decoder(latent_vec, _enthalpy, None, freerun=True)  # y [nSample, maxlength]
        numVectors = y + 2  #.argmax(dim=1) + 2 #+2是因为0和1对应起始和结束的索引，隐空间拟合时不包含这两个，采样后转换为实际分子时需补上
        numVectors_max = numVectors.argmax(dim=1)
        return numVectors.cpu(), None

    def latent_space_quality(self, nSample, tokenizer=None):
        """评估隐空间质量
        Args:
            nSample: 采样数量
            tokenizer: 分词器
        Returns:
            有效SMILES的数量
        """
        self.decoder.eval()
        # 从隐空间采样并生成分子
        numVectors, _ = self.sample(nSample)# 采样得到用于表示分子的数字序列
        # 将数字序列转换回SMILES字符串
        smilesStrs = tokenizer.getSmiles(numVectors)
        # smilesStrs = tokenizer.untokenize(numVectors)
        # # 2. 将整数索引转换为 SMILES 字符串
        # smilesStrs = []
        # for indices in numVectors:
        #     sm = tokenizer.untokenize(indices.cpu().numpy())  # 将整数索引转换为 SMILES
        #     # smilesStrs.append(sm)
        # 3. 检查每个 SMILES 的有效性
        validSmilesStrs = [sm for sm in smilesStrs if utils.isValidSmiles(sm)]
        return len(validSmilesStrs)

    def predict_enthalpy_list(self, gen_smiles, cond_enthalpy):
        # 调用模型，最好是改一下训练逻辑，就是一个batch训练完之后，收集所有生成的smiles列表，判断有效性，有效的再调用模型进行预测，统一返回预测生成焓结果
        # Placeholder for the function that predicts enthalpy from SMILES
        pass

    def calculate_enthalpy_loss(self, gen_smiles, cond_enthalpy, lb, ub):  # 需要数据集中的归一化上下限来反归一化，以和预测数据的大小匹配
        # Calculate the loss for the enthalpy prediction
        gt_enthalpy = cond_enthalpy
        predicted_enthalpy = torch.zeros(len(gen_smiles))
        valid_enthalpy = []
        _mask = torch.zeros(len(gen_smiles))
        _valid_id_mask = []

        for idx, smiles in enumerate(gen_smiles):  # 对于每个smiles，如果有效则使用predict方法，并append有效值；若无效则append 0
            if utils.isValidSmiles(smiles):  # 初步判断有效结构
                predicted_value = predict_enthalpy(smiles)
                if predicted_value != 0:  # 保存预测结果不为0的值
                    predicted_enthalpy[idx] = predicted_value  # 将生成的全0tensor中对应项保存为预测值
                    _mask[idx] = 1  # 将mask对应位置标记为有效
                    _valid_id_mask.append(idx)
                    valid_enthalpy.append(predicted_value) # 单独拎出有效结果，后面用len统计

        # 将有效的预测值和 enthalpy 转换为张量
        predicted_enthalpy_tensor = torch.tensor(predicted_enthalpy, device=self.device)
        valid_enthalpy_tensor = torch.tensor(valid_enthalpy, device=self.device)
        gt_enthalpy_tensor = torch.tensor(gt_enthalpy, device=self.device)
        gt_mask_enthalpy_tensor = torch.tensor([gt_enthalpy_tensor[i] for i in range(len(gt_enthalpy_tensor)) if _mask[i] == 1], device=self.device)
        # gt_mask_enthalpy_tensor = torch.where(_mask == 1, gt_enthalpy_tensor, torch.tensor(0.0, device=self.device))

        if len(valid_enthalpy_tensor) > 0:
            normed_valid_enthalpy_tensor = (valid_enthalpy_tensor - lb) / (ub - lb) # Normalization
            # recover_gt_mask_enthalpy_tensor = gt_mask_enthalpy_tensor * (ub - lb) + lb
            cond_loss_mean = torch.nn.functional.mse_loss(normed_valid_enthalpy_tensor, gt_mask_enthalpy_tensor)  # 只计算有效样本的损失
        else:
            cond_loss_mean = torch.tensor(1.0, device=self.device)  # 如果没有有效样本，返回1

        return cond_loss_mean  # loss_per_sample_tensor, predicted_enthalpy_tensor, valid_enthalpy_tensor

    def trainModel(self, dataloader, encoderOptimizer, decoderOptimizer, encoderScheduler, decoderScheduler, KLD_alpha,
                   nepoch, tokenizer, printInterval, lb, ub):
        self.encoder.loadState()
        self.decoder.loadState()
        self.lb = lb
        self.ub = ub
        minloss = None
        numSample = 100  # 训练过程中采样，用于计算valid数量
        for epoch in range(1, nepoch + 1):
            reconstruction_loss_list, accumulated_reconstruction_loss, kld_loss_list, accumulated_kld_loss, cond_loss_list, accumulated_cond_loss = [], 0, [], 0, [], 0
            quality_list, numValid_list = [], []
            for nBatch, (X, enthalpy) in enumerate(dataloader, 1):
                # print(f'Epoch {epoch}, batch {nBatch} is initialized, start training for this batch.')
                self.encoder.train()
                self.decoder.train()
                # 数据转移到设备
                X = X.to(self.device)  # [batch_size, maxLength] (整数索引)
                enthalpy = enthalpy.to(self.device)  # [batch_size]
                latent_vec, mu, logvar = self.encoder(X, enthalpy)  # 编码器前向(512, 64)[batch_size, latent_dim]
                pred_y = self.decoder(mu, enthalpy, X)  # [batch_size, maxLength, num_vocabs] forward(self, latent_vec, enthalpy, inp,
                checkcheck = pred_y.view(-1)
                checkcheck2 = X.view(-1)
                # 计算重构损失（交叉熵）
                reconstruction_loss = F.cross_entropy(
                    pred_y.view(-1, self.decoder.num_vocabs),  # [batch * maxLength, num_vocabs]
                    X.view(-1))  # [batch * maxLength]
                # 计算 KL 散度
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld_loss = kld_loss.mean() * KLD_alpha

                # 计算条件损失（假设已实现）
                predicted_indices = torch.argmax(pred_y, dim=-1)  # [batch_size, maxLength]
                predicted_smiles = tokenizer.getSmiles(predicted_indices)
                cond_loss_mean = self.calculate_enthalpy_loss(predicted_smiles, enthalpy, lb, ub)

                # 总损失
                if cond_loss_mean != 0 and abs(cond_loss_mean) < 10:  # 条件损失有效
                    loss = reconstruction_loss + kld_loss + cond_loss_mean
                else:
                    loss = reconstruction_loss + kld_loss

                # 反向传播
                encoderOptimizer.zero_grad()
                decoderOptimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
                loss.backward()
                encoderOptimizer.step()
                decoderOptimizer.step()

                # 记录损失
                reconstruction_loss_list.append(reconstruction_loss.item())
                kld_loss_list.append(kld_loss.item())
                cond_loss_list.append(cond_loss_mean.item())
                accumulated_reconstruction_loss += reconstruction_loss.item()
                accumulated_kld_loss += kld_loss.item()
                accumulated_cond_loss += cond_loss_mean.item()

                # 打印训练信息
                if (nBatch == 1 or nBatch % printInterval == 0):
                    # 计算重构质量
                    quality = self.reconstruction_quality_per_sample(X, enthalpy).mean()
                    quality_list.append(quality)
                    # 计算隐空间质量
                    numValid = self.latent_space_quality(numSample, tokenizer)
                    numValid_list.append(numValid)
                    print(
                        f"[{time.ctime()}] Epoch {epoch:4d} & Batch {nBatch:4d}: "
                        f"Reconstruction_Loss= {sum(reconstruction_loss_list) / len(reconstruction_loss_list):.5e} "
                        f"KLD_Loss= {sum(kld_loss_list) / len(kld_loss_list):.5e} "
                        f"Quality= {quality:.0f}/{self.decoder.maxLength} "
                        f"Valid= {numValid:.0f}/{numSample}"
                    )
                    reconstruction_loss_list.clear()
                    kld_loss_list.clear()
                    cond_loss_list.clear()
                    # 保存最佳模型
                    if minloss is None or loss.item() < minloss:
                        self.encoder.saveState()
                        self.decoder.saveState()
                        minloss = loss.item()
            encoderScheduler.step()
            decoderScheduler.step()
            # 打印 epoch 总结
            print(
                f"[{time.ctime()}] Epoch {epoch:4d}: "
                f"Reconstruction_Loss= {accumulated_reconstruction_loss / nBatch:.5e} "
                f"KLD_Loss= {accumulated_kld_loss / nBatch:.5e} "
                f"Condition_Loss= {accumulated_cond_loss / nBatch:.5e} "
                f"Quality= {sum(quality_list) / len(quality_list):.0f}/{self.decoder.maxLength} "
                f"Valid= {sum(numValid_list) / len(numValid_list):.0f}/{numSample}"
            )