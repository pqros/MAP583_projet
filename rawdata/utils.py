import torch
import pandas as pd
from rdkit.Chem import AllChem

def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    #labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    #labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    #labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)   # 0 -> train
    folds = folds.replace('Valid', 1)   # 1 -> valid
    folds = folds.replace('Test', 2)    # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, labels.values, folds.values

def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                                          rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    #labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values

def _load_clintox_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    #labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values
# input_path = 'dataset/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

def _load_muv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
       'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
       'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    #labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
       'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
       'Investigations', 'Musculoskeletal and connective tissue disorders',
       'Gastrointestinal disorders', 'Social circumstances',
       'Immune system disorders', 'Reproductive system and breast disorders',
       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
       'General disorders and administration site conditions',
       'Endocrine disorders', 'Surgical and medical procedures',
       'Vascular disorders', 'Blood and lymphatic system disorders',
       'Skin and subcutaneous tissue disorders',
       'Congenital, familial and genetic disorders',
       'Infections and infestations',
       'Respiratory, thoracic and mediastinal disorders',
       'Psychiatric disorders', 'Renal and urinary disorders',
       'Pregnancy, puerperium and perinatal conditions',
       'Ear and labyrinth disorders', 'Cardiac disorders',
       'Nervous system disorders',
       'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    #labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_toxcast_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    #labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values

def deleteNoneMol(smilelist, mollist, labellist):
    smilelist1 = []
    mollist1 = []
    labellist1 = []
    for i, mol in enumerate(mollist):
        if mol != None:
            smilelist1.append(smilelist[i])
            mollist1.append(mol)
            labellist1.append(labellist[i])
    return smilelist1, mollist1, labellist1