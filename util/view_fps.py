import pickle
import util
from rdkit import DataStructs
import numpy as np

def view_fingerprints(pkl_fname):
    """查看指纹文件的内容
    
    Args:
        pkl_fname: 指纹文件路径
    """
    print(f"Loading fingerprints from {pkl_fname}")
    with open(pkl_fname, 'rb') as f:
        fps = pickle.load(f)
    
    print(f"\nTotal fingerprints: {len(fps)}")
    print(f"Type of fingerprint: {type(fps[0])}")
    print(f"Number of bits: {fps[0].GetNumBits()}")
    
    # 随机选择两个指纹计算相似度
    if len(fps) >= 2:
        fp1, fp2 = fps[0], fps[1]
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        print(f"\nExample: Tanimoto similarity between first two molecules: {similarity:.3f}")
    
    # 计算一些统计信息
    # 将第一个指纹转换为比特数组
    bits = list(fps[0].ToBitString())
    print(f"\nFirst fingerprint bit pattern (first 64 bits):")
    print(''.join(bits[:64]))
    
    # 计算所有指纹的平均密度
    densities = [fp.GetNumOnBits() / fp.GetNumBits() for fp in fps[:1000]]  # 取前1000个样本
    avg_density = np.mean(densities)
    print(f"\nAverage fingerprint density (from 1000 samples): {avg_density:.3f}")

if __name__ == '__main__':
    pkl_fname = util.config['fname_fps']
    view_fingerprints(pkl_fname) 