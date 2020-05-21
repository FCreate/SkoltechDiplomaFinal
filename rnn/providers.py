import  re
from enum import IntEnum

from tqdm import tqdm
from rdkit import Chem
import molvs

from utils import map_parallel#, make_one_hot

molvs_standardizer = molvs.Standardizer()


class SpecialTokens(IntEnum):
    PAD = 0
    BOS = 1
    EOS = 2

class SpecialTokenWrapperModel():

    def __init__(self, model,padding):
        self.model = model
        self.padding = padding
        self.special_tokens = {0:"<pad>",1:"<bos>", 2:"<eos>"}
        self.shift = len(SpecialTokens)
        if hasattr(self.model,'vocab'): #Like SmilesModel
            self.vocabulary_size = self.model.vocsize +3
        elif hasattr(self.model,'rules'): #Like Legogram
            self.vocabulary_size = len(self.model.rules) +3

    def encode(self,seq):
        bos_seq_eos = [int(SpecialTokens.BOS)] + [s + self.shift for s in self.model.encode(seq)] + [int(SpecialTokens.EOS)]
        if self.padding:
            padded = bos_seq_eos + [int(SpecialTokens.PAD)] * (self.padding - len(bos_seq_eos))
        else:
            return bos_seq_eos

    def get_vocabulary_size(self):
        return self.vocabulary_size

    def decode(self,seq):
        seq = [s - self.shift for s in seq if s not in SpecialTokens]
        return self.model.decode(seq)


def smiles_atom_tokenizer (smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

def standardize_smile(sm):
    mol = Chem.MolFromSmiles(sm)
    mol = molvs_standardizer(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)
def robust_standardizer(smi):
    try:
        smi = standardize_smile(smi)
        return smi
    except:
        return None
def augment_smile (sm):
    mol = Chem.MolFromSmiles(sm)
    mol = molvs_standardizer(mol)
    return Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=False)

def reverse_vocab (vocab):
    return dict((v,k) for k,v in vocab.items())

class SmilesModel():
    def __init__(self, smiles, naug=3):
        vocab = set()
        for sm in tqdm(smiles):
            for n in range(naug):
                tokens = smiles_atom_tokenizer(sm)#augment_smile(sm))
                #union
                vocab |= set(tokens)
        self.vocab = {v:k for k,v in enumerate(sorted(vocab))}
        self.vocab['<unk>'] = max(self.vocab.values())
        self.inv_vocab = reverse_vocab(self.vocab)
        self.vocsize = len(self.vocab)

    def encode(self, seq):
        encoded_sm = []
        for token in smiles_atom_tokenizer(seq):
            if (token in self.vocab):
                encoded_sm.append(self.vocab[token])
            else:
                encoded_sm.append(self.vocab["<unk>"])
        return encoded_sm

    def decode(self, seq):
        return "".join([self.inv_vocab[code] for code in seq])

import torch
from torch.utils.data import Dataset

class MolecularNotationDataset(Dataset):
    def __init__(self, smiles, model, data):
        special_tokens = {v:k for k,v in enumerate(model.special_tokens)}
        self.smiles = smiles
        self.model = model
        self.vocsize = model.vocsize
        print(self.vocsize)
        self.smiles_data = [(smiles[idx], data[idx, :])for idx in range(len(smiles))]
        self.encoded = map_parallel(self.smiles_data,lambda s:([2971]+ model.encode(s[0]) + [2972], s[1]),nworkers=6)
    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return torch.LongTensor(self.encoded[idx][0])


