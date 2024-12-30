import os
import pickle
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
import util

def generate_fingerprints(smi_fname, pkl_fname):
    """生成分子指纹文件
    
    Args:
        smi_fname: SMILES文件路径
        pkl_fname: 指纹文件保存路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(pkl_fname), exist_ok=True)
    
    fps = []
    print(f"Reading SMILES from {smi_fname}")
    
    # 读取SMILES并生成指纹
    with open(smi_fname, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
        
    print(f"Generating fingerprints for {len(smiles_list)} molecules...")
    for smi in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # 使用Morgan指纹算法，半径2，1024位
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, 
                radius=2, 
                nBits=1024, 
                useFeatures=True
            )
            fps.append(fp)
    
    print(f"Generated {len(fps)} valid fingerprints")
    
    # 保存指纹
    print(f"Saving fingerprints to {pkl_fname}")
    with open(pkl_fname, 'wb') as f:
        pickle.dump(fps, f)
    
    print("Done!")

if __name__ == '__main__':
    # 从config中获取文件路径
    smi_fname = util.config['fname_dataset']
    pkl_fname = util.config['fname_fps']
    
    generate_fingerprints(smi_fname, pkl_fname) 