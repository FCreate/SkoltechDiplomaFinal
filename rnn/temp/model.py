import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import CrossEntropyLoss
from providers import SpecialTokens
from grammars.rnn_sampler import graph2mol
from torch.nn import NLLLoss

class RNN(nn.Module):

    def __init__(self,voc_size, hidden_size=512,num_layers=3,device='cuda'):
        self.device = device
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(voc_size,hidden_size)
        self.gru = nn.GRU(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=False,batch_first=True)
        self.linear = nn.Linear(hidden_size, voc_size)
        self.to(self.device)


    def forward(self, x, lengths):

        hidden = self.initHidden(x.size(0))
        #hidden = hidden.repeat(3,1,1)
        embeddings = self.embedding(x)
        input = pack_padded_sequence(embeddings,lengths,batch_first=True)
        output, hidden_out = self.gru(input, hidden)
        output, out_lengths  = pad_packed_sequence(output,batch_first=True)
        hidden_out = hidden_out[2, :, :]
        output = self.linear(output)
        output = output.transpose(1, 2)
        #hidden_out = self.hidden2endp2(self.sigmoid(self.hidden2endp1(hidden_out)))
        return output, out_lengths#, hidden_out

    def forward_to_sample(self, x,hidden):
        embeddings = self.embedding(x)
        output, hidden = self.gru(embeddings, hidden)
        output = self.linear(output)
        output.transpose_(1, 2)
        return output,hidden

    def initHidden(self,batch_size):
        return torch.zeros(self.num_layers, batch_size, 512, device=self.device)


    def sample(self, batch_size,lexical_model, max_length=140):
        print("batch size")
        print(batch_size)
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = 2971 #FixMe -- possible mistake ???

        x = start_token.cuda().unsqueeze(1)

        finished = torch.zeros(batch_size).byte()
        state = lexical_model.init_state(batch_size)
        hidden = self.initHidden(batch_size)
        #hidden = self.endp2hidden(endp)
        #out = self.forward(x)

        #print(tokens.shape)
        for step in range(max_length):
            logits,hidden = self.forward_to_sample(x,hidden)
            state,x = lexical_model.sample(state,logits.squeeze(2).cpu().detach().numpy(),0)

            x = Variable(torch.cuda.LongTensor(x)).unsqueeze(1)

        correct = [Chem.MolToSmiles(graph2mol(g)) for g, finished in zip(*state) if finished]
        return correct


    def sample_reinvent(self, batch_size, max_length=140):
        """
            Sample a batch of sequences
            Args:
                batch_size : Number of sequences to sample
                max_length:  Maximum length of the sequences
            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
            {0:"<pad>",1:"<bos>", 2:"<eos>"}
            But in real sample we should substruct 3 to equalize this.
        """
        """ """
        print("batch size")
        print(batch_size)
        start_token = Variable(torch.zeros(batch_size).long()).cuda()
        start_token[:] = 1
        h = self.initHidden(batch_size)
        x = start_token

        sequences = []
        log_probs = Variable(torch.zeros(batch_size)).cuda()
        finished = torch.zeros(batch_size).byte().cuda()
        if torch.cuda.is_available():
            finished = finished.cuda()
        loss = NLLLoss(reduction='none')
        logits_full = []
        for step in range(max_length):
            logits, h = self.forward_to_sample(x.unsqueeze(1), h)
            logits_full.append(logits)
            prob = F.softmax(logits)
            log_prob = F.log_softmax(logits)
            x = torch.multinomial(prob.squeeze(2),1).view(-1)
            sequences.append(x.view(-1, 1))
            log_probs +=  loss(log_prob.squeeze(2), x)

            x = Variable(x.data)
            EOS_sampled = (x == 2).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        logits_full = torch.cat(logits_full, 2)
        sequences = torch.cat(sequences, 1)
        return sequences.data, logits_full.data, log_probs


    def likelihood_reinvent(self, target):
        """
            Retrieves the likelihood of a given sequence
            Args:
                target: (batch_size * sequence_lenght) A batch of sequences
            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long()).cuda()
        start_token[:] = 2
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size).cuda()

        log_probs = Variable(torch.zeros(batch_size), dtype = torch.float).cuda()
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits)
            prob = F.softmax(logits)
            log_probs += NLLLoss(log_prob, target[:, step])
        return log_probs
