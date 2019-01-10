'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import modules
from typing import Dict, List, Iterator
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection

class StackedLSTM(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 input_size: int,
                 hidden_size: int, 
                 dropout: float):

        super(StackedLSTM, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, 
                input: torch.Tensor, 
                hidden: torch.Tensor) -> tuple:

        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class rnn_encoder(nn.Module):

    def __init__(self, 
                 emb_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 vocab_size: int, 
                 dropout: float, 
                 bidirectional: bool, 
                 embedding: nn.Module = None) -> None:

        super(rnn_encoder, self).__init__()
        
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.bidirectional = bidirectional

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        
        self.rnn = nn.LSTM(input_size=emb_size, 
                           hidden_size=hidden_size,
                           num_layers=num_layers, 
                           dropout=dropout, 
                           bidirectional=bidirectional,
                           batch_first=True)

    def forward(self, 
                input: torch.Tensor, 
                lengths: torch.Tensor) -> Dict:

        embs = pack(self.embedding(input), lengths, batch_first=True)

        self.rnn.flatten_parameters()
        hidden_outputs, final_state = self.rnn(embs)

        hidden_outputs = unpack(hidden_outputs, batch_first=True)[0]

        if self.bidirectional:
            hidden_outputs = hidden_outputs[:, :, :self.hidden_size] + hidden_outputs[:, :, self.hidden_size:]

        outputs = {'hidden_outputs': hidden_outputs, 'final_state': final_state}
        return outputs



class rnn_decoder(nn.Module):

    def __init__(self, 
                 emb_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 vocab_size: int, 
                 dropout: float, 
                 embedding=None):
        super(rnn_decoder, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)

        self.rnn = StackedLSTM(input_size=emb_size, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout)

        self.attention = modules.attention.luong_attention(hidden_size)


    def forward(self, 
                inputs: torch.Tensor, 
                encoder_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        embs = self.embedding(inputs)
        hidden_outputs, state, attention_weights = [], encoder_outputs['final_state'], []

        for emb in embs.split(1, dim=1):
            x = emb.squeeze(1)
            output, state = self.rnn(x, state)
            output, attn_weights = self.attention(output, x, encoder_outputs['hidden_outputs'])

            attention_weights.append(attn_weights)
            output = self.dropout(output)
            hidden_outputs.append(output)
        
        hidden_outputs = torch.stack(hidden_outputs, dim=1)
        outputs = {'hidden_outputs': hidden_outputs, 'final_state': state}

        return outputs


    def decode_step(self, 
                    input: torch.Tensor, 
                    state: torch.Tensor, 
                    contexts: torch.Tensor):
        x = self.embedding(input)
        output, state = self.rnn(x, state)
        output, attn_weigths = self.attention(output, x, contexts)

        outputs = {'hidden_output': output, 'state': state, 'attention_weights': attn_weigths}
        return outputs
