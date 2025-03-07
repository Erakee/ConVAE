import torch


class Tokenizer(object):
    def __init__(self, multiCharTokens, handleBraces=True, toolTokens=('<start>', '<end>', '<pad>')):
        self.tokensDict, self.tokensInvDict = dict(), dict()
        self.toolTokens = toolTokens
        self.handleBraces = handleBraces
        self.multiCharTokens = [t.lower() for t in sorted(multiCharTokens, key=lambda s: len(s), reverse=True)]
        for token in self.toolTokens:
            self.addToken(token)
    
    def addToken(self, token):
        if token not in self.tokensDict:
            num = len(self.tokensDict)
            self.tokensDict[token] = num
            self.tokensInvDict[num] = token

    def tokenize(self, smilesStrs, useTokenDict=False):
        vectors = []
        if useTokenDict:
            appliedMultiCharToken = sorted([key for key in self.tokensDict.keys() if key not in self.toolTokens and len(key) > 1], key=lambda s: len(s), reverse=True)
        else:
            appliedMultiCharToken = self.multiCharTokens
        for smilesStr in smilesStrs:
            currentVector = []
            startIdx, endIdx, length = 0, 0, len(smilesStr)
            foundMultiCharToken = False
            while startIdx < length:
                foundMultiCharToken = False
                if self.handleBraces and not useTokenDict and smilesStr[startIdx] == '[':
                    endIdx = smilesStr.index(']', startIdx) + 1
                    foundMultiCharToken = True
                else:
                    for token in appliedMultiCharToken:
                        tokenLen = len(token)
                        endIdx = startIdx + tokenLen
                        if endIdx > length:
                            pass
                        else:
                            if (not useTokenDict and smilesStr[startIdx: endIdx].lower() == token) or (useTokenDict and smilesStr[startIdx: endIdx] == token):
                                foundMultiCharToken = True
                                break
                if not foundMultiCharToken:
                    endIdx = startIdx + 1
                currentVector.append(smilesStr[startIdx: endIdx])
                startIdx = endIdx
            vectors.append(currentVector)
        return vectors
    
    def getTokensSize(self):
        return len(self.tokensDict)
    
    def getTokensNum(self, token):
        return self.tokensDict[token]

    def getNumVector(self, vectors, addStart=False, addEnd=False):
        numVector = []
        for vec in vectors:
            currentVec = []
            if addStart:
                currentVec.append(self.tokensDict['<start>'])
            for elem in vec:
                currentVec.append(self.tokensDict[elem])
            if addEnd:
                currentVec.append(self.tokensDict['<end>'])
            numVector.append(currentVec)
        return numVector

    # def getSmiles(self, numVectors):
    #     smileslist = []
    #     for numVector in numVectors:
    #         numVector = numVector.tolist()
    #         for i in range(len(numVector) - 1, -1, -1):
    #             if numVector[i] != 2:
    #                 break
    #             else:
    #                 numVector.pop(i)
    #         if len(numVector) > 0 and numVector[-1] == 1:
    #             numVector.pop()
    #         if len(numVector) > 0 and numVector[0] == 0:
    #             numVector.pop(0)
    #         smileslist.append(''.join([self.tokensInvDict[n] for n in numVector]))
    #     return smileslist

    def getSmiles(self, numVectors):  # 改进使用索引,去除工具 Token（如 <start>、<end>、<pad>）并生成 SMILES
        smileslist = []
        for numVector in numVectors:
            # 转换为列表（假设输入是张量）
            if isinstance(numVector, torch.Tensor):
                numVector = numVector.cpu().numpy().tolist()

            # 去除 <pad> (索引2)、<end> (索引1)、<start> (索引0)
            filtered = [
                n for n in numVector
                if n not in [self.tokensDict[t] for t in self.toolTokens]
            ]

            # 转换为 SMILES
            smi = ''.join([self.tokensInvDict[n] for n in filtered if n in self.tokensInvDict])
            smileslist.append(smi)
        return smileslist

    def untokenize(self, indices): # 直接根据索引生成 SMILES，不处理工具 Token。
        """将整数索引转换为 SMILES 字符串
        Args:
            indices: 整数索引列表（或张量）
        Returns:
            SMILES 字符串
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()  # 转换为列表
        tokens = [self.tokensInvDict.get(i, '') for i in indices]  # 处理未知索引
        return ''.join(tokens)

    def getInputNumVector(self, numVectors):
        smileslist = []
        for numVector in numVectors:
            numVector = numVector.tolist()
            for i in range(len(numVector) - 1, -1, -1):
                if numVector[i] != 2:
                    break
                else:
                    numVector.pop(i)
            if len(numVector) > 0 and numVector[-1] == 1:
                numVector.pop()
            if len(numVector) > 0 and numVector[0] == 0:
                numVector.pop(0)
            smileslist.append(''.join([self.tokensInvDict[n] for n in numVector]))
        return smileslist


def getTokenizer(file, handleBraces=True):
    tokenizer = Tokenizer(['Li', 'Be', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ca', 'As', 'Se', 'Br', 'Te', '@@'], handleBraces=handleBraces)
    with open(file, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            tokens = set(*tokenizer.tokenize([line.strip()]))
            for token in tokens:
                tokenizer.addToken(token)
    return tokenizer
