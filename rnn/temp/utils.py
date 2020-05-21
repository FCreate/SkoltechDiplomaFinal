import time
import multiprocessing as mp

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from rdkit import Chem
def map_parallel(lst, fn, nworkers=1):
    if nworkers == 1:
        return [fn(item) for item in lst]
    L = mp.Lock()
    QO = mp.Queue(nworkers)
    QI = mp.Queue()
    for item in lst:
        QI.put(item)

    def pfn():
        time.sleep(0.001)
        # print(QI.empty(), QI.qsize())
        while QI.qsize() > 0:  # not QI.empty():
            L.acquire()
            item = QI.get()
            L.release()
            obj = fn(item)
            QO.put(obj)

    procs = []
    for nw in range(nworkers):
        P = mp.Process(target=pfn, daemon=True)
        time.sleep(0.001)
        P.start()
        procs.append(P)

    return [QO.get() for i in tqdm(range(len(lst)))]


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target

"""def collect(sequences):
    #mols = [elem for elem in sequences]
    #endpoints = [elem[1] for elem in sequences]
    mols = sorted(sequences,key=len,reverse=True)
    lengths = [seq.size(0) for seq in mols]
    return pad_sequence(mols,batch_first=True), lengths"""

def collect(sequences):
    sequences = sorted(sequences, key = lambda x: x.size(), reverse =  True)
    mols = [seq for seq in sequences]
    lengths = [seq.size(0) for seq in mols]
    return pad_sequence(mols,batch_first=True), lengths

def unique_func(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))




def collate_fn(arr):
    """Function to take a list of encoded sequences and turn them into a batch"""
    max_length = max([len(seq) for seq in arr])
    collated_arr = Variable(torch.zeros(len(arr), max_length))
    for i, seq in enumerate(arr):
        collated_arr[i, :seq.size(0)] = seq
    return collated_arr

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles

def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i / len(smiles)

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))