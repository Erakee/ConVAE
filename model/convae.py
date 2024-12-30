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
            torch.nn.Linear(maxLength * num_vocabs + con_dims,
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
        X = torch.cat((X.to(self.device), enthalpy.unsqueeze(1)), dim=1)
        X = self.fc(X)
        mu, logvar = self.mu(X), self.logvar(X)
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
        self.gru = torch.nn.GRU(latent_dim + num_vocabs, hidden_dim,
                                num_hidden, batch_first=True, device=self.device)
        self.fc = torch.nn.Linear(hidden_dim, num_vocabs, device=self.device)

    def forward(self, latent_vec, enthalpy, inp, freerun=False, randomchoose=True):
        # X = latent_vec.unsqueeze(1).repeat(1, self.maxLength, 1)
        # X, _ = self.gru1(X)
        # inp_zeros = torch.zeros((inp.shape[0], 1, inp.shape[2]), dtype=torch.float32)
        # inp_new = torch.concat([inp_zeros, inp[:, :self.maxLength - 1, :]], dim=1)
        # X = torch.concat([X, inp_new], dim=2)
        # X, _ = self.gru2(X)
        latent_vec = latent_vec.to(self.device)
        enthalpy = enthalpy.unsqueeze(1).repeat(1, self.maxLength, 1)
        if not freerun:
            X = latent_vec.unsqueeze(1).repeat(1, self.maxLength, 1)
            inp_zeros = torch.zeros(
                (inp.shape[0], 1, inp.shape[2]), dtype=torch.float32, device=self.device)
            inp_new = torch.concat(
                [inp_zeros, inp[:, :self.maxLength - 1, :]], dim=1)
            X = torch.concat([X, inp_new, enthalpy], dim=2)
            X, _ = self.gru(X)
            return self.fc(X)
        else:
            out = torch.zeros(
                (latent_vec.shape[0], self.maxLength, self.num_vacabs), dtype=torch.float32)
            X_latent = latent_vec.unsqueeze(1)
            X = torch.concat([X_latent, torch.zeros(
                (latent_vec.shape[0], 1, self.num_vacabs), dtype=torch.float32, device=self.device)], dim=2)
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

    def reconstruction_quality_per_sample(self, X):
        """计算重构质量
        Args:
            X: 输入的one-hot编码分子表示
        Returns:
            diff: 重构准确度（每个位置正确重构的数量）
        """
        self.encoder.eval()
        self.decoder.eval()
        
        # 1. 通过编码器获取隐空间表示
        latent_vec, mu, logvar = self.encoder(X)
        # 2. 通过解码器重构输入
        pred_y = self.decoder(mu, X)
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
        # 通过解码器生成分子
        y = self.decoder(latent_vec, None, freerun=True)
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
    
    def predict_enthalpy_list(self, smiles):
        # 调用模型，最好是改一下训练逻辑，就是一个batch训练完之后，收集所有生成的smiles列表，判断有效性，有效的再调用模型进行预测，统一返回预测生成焓结果
        # Placeholder for the function that predicts enthalpy from SMILES
        pass

    def calculate_enthalpy_loss(self, gen_smiles, cond_enthalpy):
        # Calculate the loss for the enthalpy prediction
        predicted_enthalpy = predict_enthalpy(gen_smiles)
        return torch.nn.functional.mse_loss(predicted_enthalpy, cond_enthalpy)

    def loss_per_sample(self, pred_y, y, mu, logvar, gen_smiles, true_enthalpy):
        reconstruction_loss = torch.nn.functional.cross_entropy(
            pred_y.transpose(1, 2), y.transpose(1, 2), reduction='none').sum(dim=1)
        kld_loss = torch.sum(-0.5 * (1.0 + logvar -
                             mu.pow(2) - logvar.exp()), dim=1)
        cond_loss = self.calculate_enthalpy_loss(gen_smiles, true_enthalpy)
        return reconstruction_loss, kld_loss, cond_loss

    def trainModel(self, dataloader, encoderOptimizer, decoderOptimizer, encoderScheduler, decoderScheduler, KLD_alpha, nepoch, tokenizer, printInterval):
        self.encoder.loadState()
        self.decoder.loadState()
        minloss = None
        numSample = 10
        for epoch in range(1, nepoch + 1):
            reconstruction_loss_list, accumulated_reconstruction_loss, kld_loss_list, accumulated_kld_loss, cond_loss_list, accumulated_cond_loss = [], 0, [], 0, [], 0
            quality_list, numValid_list = [], []
            for nBatch, (X, enthalpy) in enumerate(dataloader, 1):
                self.encoder.train()
                self.decoder.train()
                X = X.to(self.device)
                enthalpy = enthalpy.to(self.device)
                latent_vec, mu, logvar = self.encoder(X, enthalpy)
                pred_y = self.decoder(latent_vec, X, enthalpy)
                predicted_indices = pred_y.argmax(dim=2) # 这里是为了将输出解码为smiles，方便后续直接使用gnn预测生成焓
                predicted_smiles = tokenizer.getSmiles(predicted_indices)
                reconstruction_loss, kld_loss, cond_loss = self.loss_per_sample(
                    pred_y, X, mu, logvar, predicted_smiles, enthalpy)
                reconstruction_mean, kld_mean, cond_mean = reconstruction_loss.mean(), kld_loss.mean() * \
                    KLD_alpha, cond_loss.mean()
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
                    quality = self.reconstruction_quality_per_sample(X).mean()
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
            print("[%s] Epoch %4d: Reconstruction_Loss= %.5e KLD_Loss= %.5e Quality= %3d/%3d Valid= %3d/%3d" % (time.ctime(), epoch, accumulated_reconstruction_loss /
                  nBatch, accumulated_kld_loss / nBatch, accumulated_cond_loss / nBatch, sum(quality_list) / len(quality_list), self.decoder.maxLength, sum(numValid_list) / len(numValid_list), numSample))

    
