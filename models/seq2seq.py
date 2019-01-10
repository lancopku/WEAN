'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import modules
import modules.rnn as rnn
from typing import List, Dict, Iterator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.models import Model
from allennlp.data import Vocabulary
from metrics import BLEU, SequenceAccuracy


class Seq2Seq(Model):

    def __init__(self, 
                 emb_size: int, 
                 hidden_size: int, 
                 enc_layers: int, 
                 dec_layers: int, 
                 dropout: float, 
                 bidirectional: bool, 
                 beam_size: int, 
                 label_smoothing: float, 
                 vocab: Vocabulary) -> None:

        super().__init__(vocab)

        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size('tokens')
        self.beam_size = beam_size
        self.label_smoothing = label_smoothing
        self._bos = self.vocab.get_token_index('@@BOS@@')
        self._eos = self.vocab.get_token_index('@@EOS@@')
        self.encoder = rnn.rnn_encoder(emb_size, hidden_size, enc_layers, self.vocab_size, dropout, bidirectional, embedding=None)
        self.decoder = rnn.rnn_decoder(emb_size, hidden_size, dec_layers, self.vocab_size, dropout, embedding=self.encoder.embedding)
        #self.generator = nn.Linear(hidden_size, self.vocab_size)
        #self.linear = nn.Linear(hidden_size, emb_size)
        #self.generator = lambda x: torch.matmul(self.linear(x), self.encoder.embedding.weight.t())
        self.generator = lambda x: torch.matmul(x, self.encoder.embedding.weight.t())
        self.accuracy = SequenceAccuracy()


    def _get_lengths(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x > 0).sum(-1)
        return lengths


    def forward(self,
                src: Dict[str, torch.Tensor],
                tgt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        src, tgt = src['tokens'], tgt['tokens']
        lengths = self._get_lengths(src)
        lengths, indices = lengths.sort(dim=0, descending=True)
        src = src.index_select(dim=0, index=indices)
        tgt = tgt.index_select(dim=0, index=indices)

        encode_outputs = self.encoder(src, lengths)
        decode_outputs = self.decoder(tgt[:, :-1], encode_outputs)
        out_logits = self.generator(decode_outputs['hidden_outputs'])
        targets = tgt[:, 1:].contiguous()
        seq_mask = (targets>0).float()

        self.accuracy(predictions=out_logits, gold_labels=targets, mask=seq_mask)
        loss = sequence_cross_entropy_with_logits(logits=out_logits, 
                                                  targets=targets,
                                                  weights=seq_mask,
                                                  average='token',
                                                  label_smoothing=self.label_smoothing)
        outputs = {'loss': loss}

        return outputs
        

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}


    def predict(self,
                src: Dict[str, torch.Tensor],
                max_decoding_step: int) -> Dict[str, torch.Tensor]:

        with torch.no_grad(): 
            if self.beam_size == 1:
                return self.greedy_search(src, max_decoding_step)
            else:
                return self.beam_search(src, max_decoding_step)


    def greedy_search(self, 
                      src: Dict[str, torch.Tensor],
                      max_decoding_step: int) -> Dict[str, torch.Tensor]:

        src = src['tokens']
        lengths = self._get_lengths(src)
        lengths, indices = lengths.sort(dim=0, descending=True)
        rev_indices = indices.sort()[1]
        src = src.index_select(dim=0, index=indices)
        bos = torch.ones(src.size(0)).long().fill_(self._bos).cuda()

        encode_outputs = self.encoder(src, lengths)
        
        inputs, state, contexts = [bos], encode_outputs['final_state'], encode_outputs['hidden_outputs']
        output_ids, attention_weights = [], []

        for i in range(max_decoding_step):
            outputs = self.decoder.decode_step(inputs[i], state, contexts)
            hidden_output, state, attn_weight = outputs['hidden_output'], outputs['state'], outputs['attention_weights']
            logits = self.generator(hidden_output)
            next_id = logits.max(1)[1]
            inputs += [next_id]
            output_ids += [next_id]
            attention_weights += [attn_weight]

        output_ids = torch.stack(output_ids, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        alignments = attention_weights.max(2)[1]
        output_ids = output_ids.index_select(dim=0, index=rev_indices)
        alignments = alignments.index_select(dim=0, index=rev_indices)
        outputs = {'output_ids': output_ids.tolist(), 'alignments': alignments.tolist()}

        return outputs


    def beam_search(self, 
                    src: Dict[str, torch.Tensor], 
                    max_decoding_step: int) -> Dict[str, torch.Tensor]:

        beam_size = self.beam_size
        src = src['tokens']
        lengths = self._get_lengths(src)
        batch_size = src.size(0)
        lengths, indices = lengths.sort(dim=0, descending=True)
        rev_indices = indices.sort()[1]
        src = src.index_select(dim=0, index=indices)

        encode_outputs = self.encoder(src, lengths)
        contexts, encState = encode_outputs['hidden_outputs'], encode_outputs['final_state']

        contexts = contexts.repeat(beam_size, 1, 1)
        decState = encState[0].repeat(1, beam_size, 1), encState[1].repeat(1, beam_size, 1)
        beam = [modules.beam.Beam(beam_size, bos = self._bos, eos = self._eos, n_best = 1)
                for _ in range(batch_size)]

        for i in range(max_decoding_step):

            if all((b.done() for b in beam)):
                break

            inp = torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(-1)

            outputs = self.decoder.decode_step(inp, decState, contexts)
            output, decState, attn = outputs['hidden_output'], outputs['state'], outputs['attention_weights']
            logits = self.generator(output)

            output = torch.nn.functional.log_softmax(logits, dim=-1).view(beam_size, batch_size, -1)
            attn = attn.view(beam_size, batch_size, -1)

            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)

        allHyps, allScores, allAttn = [], [], []

        for j in rev_indices:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        outputs = {'output_ids': allHyps, 'alignments': allAttn}
        return outputs