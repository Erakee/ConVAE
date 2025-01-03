import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import util.utils as utils
# from utils import utils
import time
from util.enthalpy_predictor import predict_enthalpy


class Encoder(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim, state_fname, device) -> None:
        super().__init__()
        self.state_fname = state_fname
        self.device = device
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(maxLength * num_vocabs + maxLength * con_dims,
                            fc_dims[0], device=self.device),
            torch.nn.ReLU()
        )
        for i in range(1, len(fc_dims)):  # [1,3)
            self.fc.append(torch.nn.Linear(
                fc_dims[i - 1], fc_dims[i], device=self.device))
            self.fc.append(torch.nn.ReLU())
        self.mu = torch.nn.Linear(fc_dims[-1], latent_dim, device=self.device)
        self.logvar = torch.nn.Linear(
            fc_dims[-1], latent_dim, device=self.device)

    def forward(self, X, enthalpy):
        enthalpy_expanded = enthalpy.unsqueeze(1).unsqueeze(2) # (512, 1, 1)
        enthalpy_expanded = enthalpy_expanded.expand(-1, X.size(1), -1) # (512, 128,1)
        X = torch.cat((X.to(self.device), enthalpy_expanded.to(self.device)), dim=2) # (512, 128,18)
        X = self.fc(X)  # (512, 128)
        mu, logvar = self.mu(X), self.logvar(X)  # (512, 64)
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


class Decoder(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, con_dims, latent_dim, hidden_dim, num_hidden, state_fname, device) -> None:
        super().__init__()
        self.state_fname = state_fname
        self.maxLength = maxLength
        self.num_vacabs = num_vocabs
        self.device = device
        # self.gru1 = torch.nn.GRU(latent_dim, hidden_dim, num_hidden - 1, batch_first=True)
        # self.gru2 = torch.nn.GRU(hidden_dim + num_vacabs, hidden_dim, 1, batch_first=True)
        self.gru = torch.nn.GRU(latent_dim + num_vocabs + con_dims, hidden_dim,
                                num_hidden, batch_first=True, device=self.device)
        self.fc = torch.nn.Linear(hidden_dim, num_vocabs, device=self.device)

    def forward(self, latent_vec, enthalpy, inp, freerun=False, randomchoose=True): # decoder(latent_vec, enthalpy, X)
        # X = latent_vec.unsqueeze(1).repeat(1, self.maxLength, 1)
        # X, _ = self.gru1(X)
        # inp_zeros = torch.zeros((inp.shape[0], 1, inp.shape[2]), dtype=torch.float32)
        # inp_new = torch.concat([inp_zeros, inp[:, :self.maxLength - 1, :]], dim=1)
        # X = torch.concat([X, inp_new], dim=2)
        # X, _ = self.gru2(X)
        latent_vec = latent_vec.to(self.device) # (512, 64)
        enthalpy = enthalpy.unsqueeze(1).unsqueeze(2).expand(-1, self.maxLength, -1) # (512, 128, 1)

        if not freerun:
            X = latent_vec.unsqueeze(1).expand(-1, self.maxLength, -1)
            inp_zeros = torch.zeros(
                (inp.shape[0], 1, inp.shape[2]), dtype=torch.float32, device=self.device)
            inp_new = torch.concat(
                [inp_zeros, inp[:, :self.maxLength - 1, :]], dim=1)
            # print(f'shape in decoder: X-latent_vec: {X.shape}, inp-origin_X: {inp_new.shape}, enthalpy: {enthalpy.shape}')
            X = torch.concat([X, inp_new, enthalpy], dim=2)
            X, _ = self.gru(X)
            return self.fc(X)
        else:
            out = torch.zeros((latent_vec.shape[0], self.maxLength, self.num_vacabs), dtype=torch.float32)
            X_latent = latent_vec.unsqueeze(1)
            X = torch.concat([X_latent, torch.zeros((latent_vec.shape[0], 1, self.num_vacabs), dtype=torch.float32, device=self.device)], dim=2)
            shift = X_latent.shape[-1]
            hidden = None
            for i in range(self.maxLength):
                y, hidden = self.gru(X, hidden)
                y = torch.nn.functional.softmax(self.fc(y), dim=-1)
                if randomchoose:
                    selected = torch.multinomial(y.squeeze(1), 1).flatten()
                else:
                    selected = torch.argmax(y.squeeze(1), dim=1)
                X[:, 0, shift:] = 0
                for j in range(len(selected)):
                    X[j, 0, shift + selected[j]] = 1
                    out[j, i, selected[j]] = 1
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
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim, hidden_dim, num_hidden, encoder_state_fname, decoder_state_fname, device) -> None:
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = Encoder(maxLength, num_vocabs, con_dims,
                               fc_dims, latent_dim, encoder_state_fname, device)
        self.decoder = Decoder(maxLength, num_vocabs, con_dims, latent_dim,
                               hidden_dim, num_hidden, decoder_state_fname, device)

    def reconstruction_quality_per_sample(self, X, enthalpy):
        """计算重构质量
        Args:
            X: 输入的one-hot编码分子表示
        Returns:
            diff: 重构准确度（每个位置正确重构的数量）
        """
        self.encoder.eval()
        self.decoder.eval()
        
        # 1. 通过编码器获取隐空间表示
        latent_vec, mu, logvar = self.encoder(X, enthalpy)
        # 2. 通过解码器重构输入
        pred_y = self.decoder(mu, enthalpy, X)
        # reconstruction_loss, kld_loss = self.loss_per_sample(pred_y, X, mu, logvar)
        
        # 3. 将预测转换为one-hot形式
        pred_one_hot = torch.zeros_like(X, dtype=X.dtype)
        pred_y_argmax = torch.nn.functional.softmax(
            pred_y, dim=2).argmax(dim=2)
        for i in range(pred_one_hot.shape[0]):
            for j in range(pred_one_hot.shape[1]):
                pred_one_hot[i, j, pred_y_argmax[i, j]] = 1
        
        # 4. 计算重构准确度
        diff = self.decoder.maxLength - \
            torch.abs(pred_one_hot - X).sum(dim=-1).sum(dim=-1) * 0.5
        return diff

    def sample(self, nSample):
        """从隐空间采样
        Args:
            nSample: 采样数量
        Returns:
            生成的token序列
        """
        # 从标准正态分布采样
        latent_vec = torch.randn(
            (nSample, self.latent_dim), device=self.device)
        _enthalpy = torch.randn(nSample, device=self.device)  # 这里随机采样的生成焓也随机生成
        # 通过解码器生成分子
        y = self.decoder(latent_vec, _enthalpy, None, freerun=True)
        numVectors = y.argmax(dim=2) + 2
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
        numVectors, _ = self.sample(nSample)
        # 将数字序列转换回SMILES字符串
        smilesStrs = tokenizer.getSmiles(numVectors)
        # 检查每个SMILES的有效性
        validSmilesStrs = []
        for sm in smilesStrs:
            if utils.isValidSmiles(sm):
                validSmilesStrs.append(sm)
        # print("ValidSmilesStrs: %s" % (validSmilesStrs,))
        return len(validSmilesStrs)
    
    def predict_enthalpy_list(self, gen_smiles, cond_enthalpy):
        # 调用模型，最好是改一下训练逻辑，就是一个batch训练完之后，收集所有生成的smiles列表，判断有效性，有效的再调用模型进行预测，统一返回预测生成焓结果
        # Placeholder for the function that predicts enthalpy from SMILES
        pass

    def calculate_enthalpy_loss(self, gen_smiles, cond_enthalpy, lb, ub): # 需要数据集中的归一化上下限来反归一化，以和预测数据的大小匹配
        # Calculate the loss for the enthalpy prediction
        predicted_enthalpy = []
        valid_enthalpy = []
        loss_per_sample = []  # 用来保存每个样本的损失值
        
        for idx, smiles in enumerate(gen_smiles):#对于每个smiles，如果有效则使用predict方法，并append有效值；若无效则append 0
            if utils.isValidSmiles(smiles):
                predicted_value = predict_enthalpy(smiles)
                predicted_enthalpy.append(predicted_value)
                valid_enthalpy.append(cond_enthalpy[idx])  # 记录有效的 enthalpy
                loss_per_sample.append(5)  # 计算有效样本时加入损失值（这里暂时设置为 0，实际损失会计算）
            else:
                # print(f"Invalid SMILES string: {smiles}, setting loss to 0.")
                predicted_enthalpy.append(0)  # 无效的 SMILES，损失记为 0
                valid_enthalpy.append(0)  # 无效的 enthalpy，损失记为 0
                loss_per_sample.append(0)  # 无效分子的损失设置为 0

        # 将有效的预测值和 enthalpy 转换为张量
        predicted_enthalpy_tensor = torch.tensor(predicted_enthalpy, device=self.device)
        valid_enthalpy_tensor = torch.tensor(valid_enthalpy, device=self.device)

        # 将每个样本的损失值保存在 loss_per_sample 中
        loss_per_sample_tensor = torch.tensor(loss_per_sample, device=self.device)
        # 计算有效的损失，只对有效的样本计算
        # 如果有有效的样本，则计算条件损失的平均值
        # valid_mask = float(valid_enthalpy != 0)#.float()  # 有效分子的mask
        valid_mask = (valid_enthalpy_tensor != 0).to(torch.float32)
        num_valid = valid_mask.sum()
        if num_valid > 0:
            valid_enthalpy_tensor = valid_enthalpy_tensor * (ub - lb) + lb  # Reverse normalization
            cond_loss_mean = torch.nn.functional.mse_loss(predicted_enthalpy_tensor[valid_mask != 0], valid_enthalpy_tensor[valid_mask != 0])  # 只计算有效样本的损失
        else:
            valid_enthalpy_tensor = torch.tensor([0.0] * len(valid_enthalpy_tensor), device=self.device)
            cond_loss_mean = torch.tensor(1.0, device=self.device)  # 如果没有有效样本，返回1

        return cond_loss_mean # loss_per_sample_tensor, predicted_enthalpy_tensor, valid_enthalpy_tensor

    def loss_per_sample(self, pred_y, y, mu, logvar, gen_smiles, true_enthalpy, lb, ub):
        reconstruction_loss = torch.nn.functional.cross_entropy( # 都是(512, 128, 17)
            pred_y.transpose(1, 2), y.transpose(1, 2), reduction='none').sum(dim=1) # 输出结果是个tensor(512)
        kld_loss = torch.sum(-0.5 * (1.0 + logvar -
                             mu.pow(2) - logvar.exp()), dim=1) # 输出结果是个tensor(512)
        cond_loss_mean = self.calculate_enthalpy_loss(gen_smiles, true_enthalpy, lb, ub)
        return reconstruction_loss, kld_loss, cond_loss_mean

    def trainModel(self, dataloader, encoderOptimizer, decoderOptimizer, encoderScheduler, decoderScheduler, KLD_alpha, nepoch, tokenizer, printInterval, lb, ub):
        self.encoder.loadState()
        self.decoder.loadState()
        self.lb = lb
        self.ub = ub
        minloss = None
        numSample = 10
        for epoch in range(1, nepoch + 1):
            reconstruction_loss_list, accumulated_reconstruction_loss, kld_loss_list, accumulated_kld_loss, cond_loss_list, accumulated_cond_loss = [], 0, [], 0, [], 0
            quality_list, numValid_list = [], []
            for nBatch, (X, enthalpy) in enumerate(dataloader, 1):
                print(f'Epoch {epoch}, batch {nBatch} is initialized, start training for this batch.')
                self.encoder.train()
                self.decoder.train()
                X = X.to(torch.float32)  # 将 X 转换为 float32
                enthalpy = enthalpy.to(torch.float32)
                X = X.to(self.device)
                enthalpy = enthalpy.to(self.device)
                latent_vec, mu, logvar = self.encoder(X, enthalpy) # (512, 64)
                print('Encoder processed!')
                pred_y = self.decoder(latent_vec, enthalpy, X) # (512, 128,17)
                print('Decoder processed!')
                predicted_indices = pred_y.argmax(dim=2)  #(512, 128) 这里是为了将输出解码为smiles，方便后续直接使用gnn预测生成焓
                predicted_smiles = tokenizer.getSmiles(predicted_indices) # list 512
                print('Decoder output decode to smiles!')
                reconstruction_loss, kld_loss, cond_loss_mean = self.loss_per_sample(
                    pred_y, X, mu, logvar, predicted_smiles, enthalpy, lb, ub)
                reconstruction_mean, kld_mean, cond_mean = reconstruction_loss.mean(), kld_loss.mean() * \
                                                            KLD_alpha, cond_loss_mean
                loss = reconstruction_mean + kld_mean + cond_mean
                encoderOptimizer.zero_grad()
                decoderOptimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
                loss.backward()
                encoderOptimizer.step()
                decoderOptimizer.step()
                reconstruction_loss_list.append(reconstruction_mean.item())
                kld_loss_list.append(kld_mean.item())
                cond_loss_list.append(cond_mean.item())
                accumulated_reconstruction_loss += reconstruction_mean.item()
                accumulated_kld_loss += kld_mean.item()
                accumulated_cond_loss += cond_mean.item()
                if (nBatch == 1 or nBatch % printInterval == 0):
                    quality = self.reconstruction_quality_per_sample(X, enthalpy).mean()
                    numValid = self.latent_space_quality(numSample, tokenizer)
                    quality_list.append(quality)
                    numValid_list.append(numValid)
                    print("[%s] Epoch %4d & Batch %4d: Reconstruction_Loss= %.5e KLD_Loss= %.5e Quality= %3d/%3d Valid= %3d/%3d" % (time.ctime(), epoch, nBatch, sum(
                        reconstruction_loss_list) / len(reconstruction_loss_list), sum(kld_loss_list) / len(kld_loss_list), quality, self.decoder.maxLength, numValid, numSample))
                    reconstruction_loss_list.clear()
                    kld_loss_list.clear()
                    cond_loss_list.clear()
                    if minloss is None:
                        minloss = loss.item()
                    elif loss.item() < minloss:
                        self.encoder.saveState()
                        self.decoder.saveState()
                        minloss = loss.item()
            encoderScheduler.step()
            decoderScheduler.step()
            print("[%s] Epoch %4d: Reconstruction_Loss= %.5e KLD_Loss= %.5e Condition_Loss=%.5e Quality= %3d/%3d Valid= %3d/%3d" % (time.ctime(), epoch, accumulated_reconstruction_loss / nBatch, accumulated_kld_loss / nBatch, accumulated_cond_loss / nBatch, sum(quality_list) / len(quality_list), self.decoder.maxLength, sum(numValid_list) / len(numValid_list), numSample))

    
