import pandas as pd
import os

def csv_to_smi(csv_path, smi_path, smiles_column='smiles'):
    """将CSV文件转换为SMILES文件
    
    Args:
        csv_path: CSV文件路径
        smi_path: 输出的.smi文件路径
        smiles_column: CSV中SMILES列的列名
    """
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 确保SMILES列存在
    if smiles_column not in df.columns:
        raise ValueError(f"CSV文件中没有找到'{smiles_column}'列")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(smi_path), exist_ok=True)
    
    # 提取SMILES并保存
    smiles_list = df[smiles_column].dropna().tolist()
    
    print(f"Found {len(smiles_list)} valid SMILES strings")
    
    # 保存为.smi文件
    with open(smi_path, 'w') as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")
    
    print(f"Saved SMILES to {smi_path}")

# 使用示例
if __name__ == '__main__':
    csv_path = 'D:/Project/genSmiles/dataset/train.csv'  # 你的CSV文件路径
    smi_path = 'D:/Project/genSmiles/dataset/train.smi'  # 输出的.smi文件路径
    
    csv_to_smi(csv_path, smi_path, smiles_column='smiles') 