import torch
import pandas as pd
import numpy as np
from util.tokens import Tokenizer
import util.utils as utils
import torch
from util import utils


def preprocess(df):
    # 生成焓数据归一化，将上下限扩大到设定阈值
    enthalpy = np.array(df)
    threshold = utils.config['threshold']
    min_enthalpy = np.min(enthalpy)
    max_enthalpy = np.max(enthalpy)
    diff = max_enthalpy - min_enthalpy
    lower_bound = min_enthalpy - threshold * diff
    upper_bound = max_enthalpy + threshold * diff
    max_diff = upper_bound - lower_bound
    normalized_enthalpy = (enthalpy - lower_bound) / (upper_bound - lower_bound)
    # normalized_enthalpy = normalized_enthalpy.to(torch.float32)
    normalized_enthalpy = torch.tensor(normalized_enthalpy, dtype=torch.float32)
    return normalized_enthalpy.tolist(), lower_bound, upper_bound


class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, fname, tokenizer, maxLength):
        super().__init__()
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.data = pd.read_csv(fname)
        self.smiles = self.data['smiles'].tolist()
        self.enthalpy = self.data['heat_of_formation'].tolist()
        self.enthalpy, self.lb, self.ub = preprocess(self.enthalpy)
        # 存储 one-hot 编码结果
        self.smiles_hot = self.one_hot_collate_fn(self.smiles)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles_hot[i], self.enthalpy[i]

    def _getbound(self):
        return self.lb, self.ub

    def collate_fn(self, smilesStrs):
        tokenVectors = self.tokenizer.tokenize(smilesStrs, useTokenDict=True)
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in self.tokenizer.getNumVector(tokenVectors, addStart=True)], padding_value=self.tokenizer.getTokensNum('<pad>'), batch_first=True), torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in self.tokenizer.getNumVector(tokenVectors, addEnd=True)], padding_value=self.tokenizer.getTokensNum('<pad>'), batch_first=True)

    def one_hot_collate_fn(self, smilesStrs):
        skip_vocab = 2
        tokenVectors = self.tokenizer.tokenize(smilesStrs, useTokenDict=True)
        numVectors = self.tokenizer.getNumVector(tokenVectors)
        one_hot_code = torch.zeros((len(smilesStrs), self.maxLength, self.tokenizer.getTokensSize() - skip_vocab), dtype=torch.float32)
        for i, vec in enumerate(numVectors):
            for j, n in enumerate(vec):
                one_hot_code[i, j, n - 2] = 1
            if j + 1 < self.maxLength:
                one_hot_code[i, j + 1:, 0] = 1
        return one_hot_code    


class SmilesDictDataset(torch.utils.data.Dataset):
    def __init__(self, fname, tokenizer, maxLength):
        super().__init__()
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.pad_idx = tokenizer.getTokensNum('<pad>')

        self.data = pd.read_csv(fname)
        self.smiles = self.data['smiles'].tolist()
        self.enthalpy = self.data['heat_of_formation'].tolist()
        self.enthalpy, self.lb, self.ub = preprocess(self.enthalpy)
        self.smiles_indices = self._preprocess_smiles()
        # 将 processed 列表转换为一个大的张量
        # self.smiles_indices = torch.stack(self.smiles_indices)
        # self.smiles_indices = self.smiles_indices.to(torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles_indices[i], torch.tensor(self.enthalpy[i])

    def _getbound(self):
        return self.lb, self.ub

    def _preprocess_smiles(self):
        """将 SMILES 转换为整数索引序列并填充"""
        processed = []
        for smi in self.smiles:
            # Step 1: Tokenize SMILES
            token_vector = self.tokenizer.tokenize(
                [smi],
                useTokenDict=True
            )[0]  # 获取第一个（唯一）SMILES的token列表

            # Step 2: 转换为整数索引
            num_vector = self.tokenizer.getNumVector(
                [token_vector],
                addStart=True,
                addEnd=True
            )[0]  # 假设需要添加 <start> 和 <end>

            # Step 3: 截断/填充到 maxLength
            # 检查索引不超过词表大小
            if max(num_vector) >= self.tokenizer.getTokensSize():
                print(f"Warning: token index {max(num_vector)} exceeds vocabulary size {self.tokenizer.getTokensSize()}")
            
            # 截断/填充到 maxLength
            if len(num_vector) > self.maxLength:
                # 保留 <start> 和 <end> 的情况下截断中间部分
                truncated = [num_vector[0]] + num_vector[1:-1][:self.maxLength - 2] + [num_vector[-1]]
                num_vector = truncated
            else:
                # 填充到 maxLength（在末尾添加 <pad>）
                padding = [self.pad_idx] * (self.maxLength - len(num_vector))
                num_vector = num_vector + padding

            processed.append(torch.tensor(num_vector))
        return processed

    # def collate_fn(self, smilesStrs):
    #     tokenVectors = self.tokenizer.tokenize(smilesStrs, useTokenDict=True)
    #     return torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in self.tokenizer.getNumVector(tokenVectors, addStart=True)],
    #                                            padding_value=self.tokenizer.getTokensNum('<pad>'),
    #                                            batch_first=True), \
    #            torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in self.tokenizer.getNumVector(tokenVectors, addEnd=True)],
    #                                            padding_value=self.tokenizer.getTokensNum('<pad>'),
    #                                            batch_first=True)

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        smiles_indices, enthalpies = zip(*batch)

        # 转换为张量 (已经预先填充到相同长度)
        smiles_tensor = torch.stack(smiles_indices)  # [batch_size, maxLength]
        enthalpies_tensor = torch.stack(enthalpies)  # [batch_size]

        return smiles_tensor, enthalpies_tensor
# 使用方法
# tokenizer = utils.get_tokenizer()
# print(tokenizer.tokensDict)
# smilesDataset = SmilesDictDataset(
#         utils.config['fname_dataset'], tokenizer, utils.config['maxLength'])
# # smiles_dataset = SmilesDataset(fname=utils.config['fname_dataset'], tokenizer=tokenizer, maxLength=utils.config['maxLength'])
# dataloader = torch.utils.data.DataLoader(
#         smilesDataset, batch_size=utils.config['batch_size'],
#         shuffle=True, num_workers=4,
#         collate_fn=smilesDataset.collate_fn)
# for batch in dataloader:
#    X, enthalpy = batch