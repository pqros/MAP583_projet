import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem

try:
    import rdkit
    from rdkit import Chem
    from rdkit import rdBase
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.rdchem import BondType as BT

    rdBase.DisableLog('rdApp.error')
except ImportError:
    rdkit = None
import os
import os.path as osp
from torch_sparse import coalesce
import torch
import torch.nn.functional as F
import rawdata.utils as utils
import rawdata.splitters as splitters
import pickle


def smile2feat(mollist):
    data_list = []
    label_list = []
    # atoms and bonds type in original file
    # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bond_type = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    data_list = []
    for i, mol in enumerate(mollist):
        if mol is None:
            continue
        # text used to get position of atoms and distance betweens atoms, we don't have it
        # text = suppl.GetItemText(i)
        N = mol.GetNumAtoms()

        # type_idx = []
        atomic_number = []
        acceptor = []
        donor = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        num_hs = []
        for atom in mol.GetAtoms():
            # type_idx.append(types[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            donor.append(0)
            acceptor.append(0)
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

        feats = factory.GetFeaturesForMol(mol)
        for j in range(0, len(feats)):
            if feats[j].GetFamily() == 'Donor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    donor[k] = 1
            elif feats[j].GetFamily() == 'Acceptor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    acceptor[k] = 1

        # x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
        x2 = torch.tensor([
            atomic_number, acceptor, donor, aromatic, sp, sp2, sp3, num_hs
        ], dtype=torch.float).t().contiguous()
        # x = torch.cat([x1.to(torch.float), x2], dim=-1)
        x = x2

        if len(mol.GetBonds()) > 0:
            row, col, bond_idx = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [bond_type[bond.GetBondType()]]
            edge_index = torch.tensor([row, col], dtype=torch.long)

            edge_attr = F.one_hot(torch.tensor(bond_idx), num_classes=len(bond_type)).to(torch.float)

            edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.float)
            edge_attr = torch.empty((0, len(bond_type)), dtype=torch.float)

        data = {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}
        data_list.append(data)
    return data_list


# In[7]:


def feat2graph(featlist):
    graphs = []
    for feat in featlist:
        edge_index = feat['edge_index']
        edge_attr = feat['edge_attr']

        n = feat['x'].shape[0]
        graph = np.empty((n, n, 13))
        for i in range(8):
            graph[:, :, i] = np.diag(feat['x'][:, i])

        affinity = np.zeros((n, n))
        edge_features = np.zeros((n, n, 4))
        for ii in range(edge_index.shape[1]):
            affinity[edge_index[0, ii], edge_index[1, ii]] = 1
            edge_features[edge_index[0, ii], edge_index[1, ii]] = edge_attr[ii]

        graph[:, :, 8] = affinity
        graph[:, :, 9:] = edge_features
        graphs.append(graph)
    graphs = np.array(graphs)
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1])
    return graphs


setlist = ['clintox', 'bace', 'bbbp', 'hiv', 'muv', 'sider', 'tox21', 'toxcast']

for dataset in setlist:
    path_to_data = './rawdata/'
    csvfile = path_to_data + dataset + '.csv'
    print(csvfile)

    if dataset == 'bace':
        smilelist, mollist, labellist, foldvalues = utils._load_bace_dataset(csvfile)
        train_idx = np.where(foldvalues == 0)
        valid_idx = np.where(foldvalues == 1)
        test_idx = np.where(foldvalues == 2)
    else:
        if dataset == 'bbbp':
            smilelist, mollist, labellist = utils._load_bbbp_dataset(csvfile)
        if dataset == 'clintox':
            smilelist, mollist, labellist = utils._load_clintox_dataset(csvfile)
        if dataset == 'hiv':
            smilelist, mollist, labellist = utils._load_hiv_dataset(csvfile)
        if dataset == 'muv':
            smilelist, mollist, labellist = utils._load_muv_dataset(csvfile)
        if dataset == 'sider':
            smilelist, mollist, labellist = utils._load_sider_dataset(csvfile)
        if dataset == 'tox21':
            smilelist, mollist, labellist = utils._load_tox21_dataset(csvfile)
        if dataset == 'toxcast':
            smilelist, mollist, labellist = utils._load_toxcast_dataset(csvfile)

        smilelist, mollist, labellist = utils.deleteNoneMol(smilelist, mollist, labellist)
        train_idx, valid_idx, test_idx = splitters.scaffold_split(smilelist, frac_train=0.8, frac_valid=0.1,
                                                                  frac_test=0.1)

    featlist = smile2feat(mollist)
    graphs = feat2graph(featlist)
    pickle.dump([graphs, labellist, train_idx, valid_idx, test_idx], open('graph_{}.pkl'.format(dataset), 'wb'))
