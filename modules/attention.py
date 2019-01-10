import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class luong_attention(nn.Module):

    def __init__(self, hidden_size, activation=None):
        super(luong_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, 
                h: torch.Tensor, 
                x: torch.Tensor, 
                contexts: torch.Tensor) -> tuple:

        gamma_h = self.linear_in(h).unsqueeze(2)    # batch * size * 1
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)

        weights = torch.bmm(contexts, gamma_h).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), contexts).squeeze(1) # batch * size
        output = self.tanh(self.linear_out(torch.cat([c_t, h], 1)))

        return output, weights




class bahdanau_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, activation=None):
        super(bahdanau_attention, self).__init__()
        self.linear_encoder = nn.Linear(hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        self.linear_r=  nn.Linear(hidden_size*2+emb_size, hidden_size*2)
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, h, x, contexts):
        gamma_encoder = self.linear_encoder(contexts)           # batch * time * size
        gamma_decoder = self.linear_decoder(h).unsqueeze(1)    # batch * 1 * size
        weights = self.linear_v(self.tanh(gamma_encoder+gamma_decoder)).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time

        c_t = torch.bmm(weights.unsqueeze(1), contexts).squeeze(1) # batch * size
        r_t = self.linear_r(torch.cat([c_t, h, x], dim=1))
        output = r_t.view(-1, self.hidden_size, 2).max(2)[0]
        return output, weights
