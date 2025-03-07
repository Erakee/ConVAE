import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mpnn import MPNNModel
import util.utils as utils
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = MPNNModel(num_layers=2, emb_dim=256, in_dim=42, edge_dim=10, out_dim=1).to(device)
model.load_state_dict(torch.load('D:/Project/VAE_Related/ConVAE/enthalpy/stateGNN.pt'))  # Adjust the path as necessary
# model.load_state_dict(torch.load('D:/Project/VAE_Related/ConVAE/enthalpy/last_epoch_graph_data.pt'))
model.eval()


def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding


def get_atom_features(atom,
                      use_chirality=True,
                      hydrogens_implicit=True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    原子类型、重原子邻居的数量、形式电荷、杂化类型、原子是否在环中、原子是否是芳香族、原子质量、范德华半径和共价半径。
    原子序号、度考虑现在补充  自由基可以后续看情况
    电子属性（该原子提供电子还是接收电子）可以看情况补充
    原子序号:{atom.GetAtomicNum()},  ###这个有必要补充一下
    手性信息:{atom.GetChiralTag()}, 
    度:{atom.GetTotalDegree()},  ###这个有必要补充一下
    电荷:{atom.GetFormalCharge()}, 
    连接氢原子数:{atom.GetTotalNumHs()}, 
    自由基:{atom.GetNumRadicalElectrons()}, #### 这个暂时应该用不到
    杂化类型:{atom.GetHybridization()}, 
    芳香性:{atom.GetIsAromatic()}, 
    是否在环上:{atom.IsInRing()
    """
    # define list of permitted atoms   
    permitted_list_of_atoms = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'Unknown']

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    # compute atom features  
    # atom_Num_env = one_hot_encoding(str(atom.GetAtomicNum()), [1, 6, 7, 8, 9, "other"])
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    degree_enc = one_hot_encoding(int(atom.GetTotalDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])  # yes
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "OTHER"])  # yes
    # degree_enc = one_hot_encoding(str(atom.GetTotalDegree()), [0, 1, 2, 3, 4, 5, 6, 7, 8])
    is_in_a_ring_enc = [int(atom.IsInRing())]  # yes
    is_aromatic_enc = [int(atom.GetIsAromatic())]  # yes
    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]
    atom_feature_vector = atom_type_enc + formal_charge_enc + hybridisation_type_enc + degree_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond,
                      use_stereochemistry=True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    键特征有：键类型、键是否共轭、键是否在环中。作为附加选项，用户可以指定是否在双键周围包含 E-Z 立体化学特征。
    可以补充距离矩阵（原子之间的最短距离）
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def create_single_graph_from_smiles(smiles, label=0.0):
    """
    Takes a single SMILES string and returns a PyTorch Geometric Data object 
    representing the molecular graph along with the associated label. 
    If no label is provided, the default label is 0.0.
    """
    # convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2 * mol.GetNumBonds()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

    # construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)
    X = torch.tensor(X, dtype=torch.float)

    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim=0)

    # construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
    for k, (i, j) in enumerate(zip(rows, cols)):
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
    EF = torch.tensor(EF, dtype=torch.float)

    # ensure label is a float (default is 0.0)
    y_tensor = torch.tensor(np.array([label]), dtype=torch.float)

    # create PyTorch Geometric Data object
    data = Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor)

    return data


def predict_enthalpy(smiles_str):
    """
    Predicts the generated enthalpy for a given SMILES string.
    Args:
        smiles_str (str): The SMILES representation of the molecule.
    Returns:
        float: The predicted enthalpy value or None if invalid.
    """
    # Check if the SMILES string is valid
    if not smiles_str or not utils.isValidSmiles(smiles_str) or len(smiles_str) <= 2:
        # print(f"Invalid SMILES string: {smiles_str}")
        return 0.0  # 或者返回一个默认值

    # Convert SMILES to graph data
    data = create_single_graph_from_smiles(smiles_str)  # 使用转换方法生成 Data 对象
    # print(f"Data: {data}")  # 打印数据以检查
    # Prepare the data for the model
    data = data.to(device)  # Move data to the appropriate device
    # Make prediction
    with torch.no_grad():
        pred = model(data)  # 直接传递 data 对象
        return pred.item()  # Return the predicted enthalpy value


# Example usage:
# smiles_exp = '[O-][N+](=O)C1=CC=NO1'  # 单个字符会报错，如'C'
# enthalpy = predict_enthalpy(smiles_exp)
# print(f"Predicted Enthalpy: {enthalpy}")
