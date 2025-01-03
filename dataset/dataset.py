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

# 使用方法
# tokenizer = utils.get_tokenizer()
# print(tokenizer.tokensDict)
# smilesDataset = SmilesDataset(
#         utils.config['fname_dataset'], tokenizer, utils.config['maxLength'])
# # smiles_dataset = SmilesDataset(fname=utils.config['fname_dataset'], tokenizer=tokenizer, maxLength=utils.config['maxLength'])
# dataloader = torch.utils.data.DataLoader(
#         smilesDataset, batch_size=utils.config['batch_size'],
#         shuffle=True, num_workers=4,
#         collate_fn=smilesDataset.one_hot_collate_fn)
# for batch in dataloader:
#    X, enthalpy = batch