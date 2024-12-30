import torch
import pandas as pd

class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, fname, tokenizer, maxLength):
        super().__init__()
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.data = pd.read_csv(fname)
        self.smiles = self.data['smiles'].tolist()
        self.enthalpy = self.data['enthalpy'].tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i], self.enthalpy[i]

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
#  smiles_dataset = SmilesDataset(fname=fname, tokenizer=tokenizer, maxLength=maxLength)
# dataloader = DataLoader(smiles_dataset, batch_size=32, shuffle=True, collate_fn=smiles_dataset.one_hot_collate_fn)
# for batch in dataloader:
#    X, enthalpy = batch