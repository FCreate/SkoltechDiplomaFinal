import time
import multiprocessing as mp

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

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

def collect(sequences):
    sequences = sorted(sequences,key=len,reverse=True)
    lengths = [seq.size(0) for seq in sequences]
    return pad_sequence(sequences,batch_first=True),lengths

