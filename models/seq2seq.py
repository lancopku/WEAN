import torch
import torch.nn as nn
from torch.autograd import Variable
import data.dict as dict
import models


class seq2seq(nn.Module):

    def __init__(self, config, src_vocab_size, tgt_vocab_size, use_cuda, pretrain=None, score_fn=None):
        super(seq2seq, self).__init__()
        if pretrain is not None:
            src_embedding = pretrain['src_emb']
            tgt_embedding = pretrain['tgt_emb']
        else:
            src_embedding = None
            tgt_embedding = None
        self.encoder = models.rnn_encoder(config, src_vocab_size, embedding=src_embedding)
        if config.shared_vocab == False:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, embedding=tgt_embedding, score_fn=score_fn)
        else:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, embedding=self.encoder.embedding, score_fn=score_fn)
        self.use_cuda = use_cuda
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        self.criterion = models.criterion(tgt_vocab_size, use_cuda)

    def compute_loss(self, hidden_outputs, targets, memory_efficiency):
        if memory_efficiency:
            return models.memory_efficiency_cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)
        else:
            return models.cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)

    def forward(self, src, src_len, tgt, tgt_len):
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        tgt = torch.index_select(tgt, dim=1, index=indices)

        contexts, state = self.encoder(src, lengths.data.tolist())
        outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1))
        return outputs, tgt[1:]

    def sample(self, src, src_len, tgt, tgt_len):

        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)
        bos = Variable(torch.ones(src.size(1)).long().fill_(dict.BOS), volatile=True)

        if self.use_cuda:
            bos = bos.cuda()

        contexts, state = self.encoder(src, lengths.tolist())
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts.transpose(0, 1))
        _, attns_weight = final_outputs
        alignments = attns_weight.max(2)[1]
        sample_ids = torch.index_select(sample_ids.data, dim=1, index=ind)
        alignments = torch.index_select(alignments.data, dim=1, index=ind)
        targets = tgt[1:]

        return sample_ids.t(), targets.t(), alignments.t()
