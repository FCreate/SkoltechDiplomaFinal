import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from grammars.grammar import graph2mol
from rdkit import Chem
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from providers import SpecialTokens


class RNN(nn.Module):

    def __init__(self,voc_size,hidden_size=512,num_layers=3,device='cuda'):
        self.device = device
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(voc_size,hidden_size)
        self.gru = nn.GRU(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=False,batch_first=True)
        self.linear = nn.Linear(hidden_size, voc_size)
        self.to(self.device)

    def forward(self, x,lengths):
        hidden = self.initHidden(x.size(0))
        embeddings = self.embedding(x)
        input = pack_padded_sequence(embeddings,lengths,batch_first=True)
        output, _ = self.gru(input, hidden)
        output,_ = pad_packed_sequence(output,batch_first=True)
        output = self.linear(output)
        output = output.transpose(1, 2)
        return output

    def forward_to_sample(self, x,hidden):
        embeddings = self.embedding(x)
        output, hidden = self.gru(embeddings, hidden)
        output = self.linear(output)
        output.transpose_(1, 2)
        return output,hidden

    def initHidden(self,batch_size):
        return torch.zeros(self.num_layers, batch_size, 512, device=self.device)


    def sample(self, batch_size,lexical_model, max_length=140):

        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = 2971 #FixMe -- possible mistake ???

        x = start_token.cuda().unsqueeze(1)

        finished = torch.zeros(batch_size).byte()
        state = lexical_model.init_state(batch_size)
        hidden = self.initHidden(batch_size)
        #out = self.forward(x)

        for step in range(max_length):
            logits,hidden = self.forward_to_sample(x,hidden)
            state,x = lexical_model.sample(state,logits.squeeze(2).cpu().detach().numpy(),0)
            x = Variable(torch.cuda.LongTensor(x)).unsqueeze(1)
        correct = [Chem.MolToSmiles(graph2mol(g)) for g, finished in zip(*state) if finished]
        return correct
