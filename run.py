from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import os
import time
import argparse
import json
from pathlib import Path
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator, BasicIterator
from trainer import Trainer
from allennlp.nn import util
from allennlp.common.tqdm import Tqdm
from metrics import SequenceAccuracy, calc_bleu_score
from models.seq2seq import Seq2Seq
from predictor import Predictor

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-emb_size', type=int, default=256, help="Embedding size")
parser.add_argument('-hidden_size', type=int, default=256, help="Hidden size")
parser.add_argument('-enc_layers', type=int, default=2, help="Number of encoder layer")
parser.add_argument('-dec_layers', type=int, default=2, help="Number of decoder layer")
parser.add_argument('-batch_size', type=int, default=64, help="Batch size")
parser.add_argument('-beam_size', type=int, default=10, help="Beam size")
parser.add_argument('-vocab_size', type=int, default=50000, help="Vocabulary size")
parser.add_argument('-epoch', type=int, default=100, help="Number of epoch")
parser.add_argument('-report', type=int, default=500, help="Number of report interval")
parser.add_argument('-lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('-lr_decay', type=float, default=1.0, help="Learning rate Decay")
parser.add_argument('-ema_decay', type=float, default=1.000, help="Moving Average rate Decay")
parser.add_argument('-dropout', type=float, default=0.4, help="Dropout rate")
parser.add_argument('-label_smoothing', type=float, default=0.0, help="Dropout rate")
parser.add_argument('-restore', type=str, default='', help="Restoring model path")
parser.add_argument('-mode', type=str, default='train', help="Train or test")
parser.add_argument('-dir', type=str, default='', help="Checkpoint directory")
parser.add_argument('-max_len', type=int, default=50, help="Limited length for text")
parser.add_argument('-max_step', type=int, default=50, help="Max decoding step")
parser.add_argument('-gpu', type=int, default=0, help="GPU device")
parser.add_argument('-lazy', action='store_true', help="Lazyness of dataset")
parser.add_argument('-bidirectional', action='store_true', help="Bidirectional model")

opt = parser.parse_args()
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.set_device(opt.gpu)

data_path = os.path.expanduser('/data/data-simplification/wikismall/')

train_path, dev_path = os.path.join(data_path, 'train.jsonl'), os.path.join(data_path, 'dev.jsonl')
test_path = os.path.join(data_path, 'test.jsonl')
ner_path = os.path.join(data_path, 'aner.json')

vocab_dir = os.path.join(data_path, 'dicts-{0}'.format(opt.vocab_size))

if opt.dir == '':
    save_dir = Path(data_path) / 'log' / time.strftime("%Y-%m-%dT%H_%M_%S")
else:
    save_dir = Path(data_path) / 'log' / opt.dir
save_dir.mkdir(parents=True, exist_ok=True)
save_dir = str(save_dir)


class PWKPReader(DatasetReader):
    """
    DatasetReader for Bookcorpus data, one sentence per line, like
        {"short_text": "chapter 20", "summary": "ch 20"}
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=opt.lazy)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, src: List[Token], tgt: List[Token]) -> Instance:
        src_field = TextField(src, self.token_indexers)
        tgt_field = TextField(tgt, self.token_indexers)
        fields = {"src": src_field, "tgt": tgt_field}
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = json.loads(line)
                src, tgt = pairs['source'].split(' '), pairs['target'].split(' ')
                if opt.max_len > 0:
                    src = src[:opt.max_len]
                    tgt = tgt[:opt.max_len]
                tgt = ['@@BOS@@'] + tgt + ['@@EOS@@']
                src = [Token(word) for word in src]
                tgt = [Token(word) for word in tgt]
                yield self.text_to_instance(src, tgt)
    
    def read_raw(self, file_path: str) -> Iterator[Dict]:
        with open(file_path) as f:
            for line in f:
                pairs = json.loads(line)
                yield pairs



def train():
    reader = PWKPReader()
    train_dataset = reader.read(train_path)
    valid_dataset = reader.read(dev_path)
    if os.path.exists(vocab_dir):
        vocab = Vocabulary.from_files(vocab_dir)
    else:
        vocab = Vocabulary.from_instances(instances=train_dataset,
                                          max_vocab_size=opt.vocab_size)
        vocab.save_to_files(vocab_dir)
    iterator = BucketIterator(batch_size=opt.batch_size, sorting_keys=[("src", "num_tokens"), ("tgt", "num_tokens")])
    iterator.index_with(vocab)

    model = Seq2Seq(emb_size=opt.emb_size,
                    hidden_size=opt.hidden_size,
                    enc_layers = opt.enc_layers,
                    dec_layers = opt.dec_layers,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                    beam_size=opt.beam_size,
                    label_smoothing=opt.label_smoothing,
                    vocab=vocab)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    #learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=opt.lr_decay)

    val_iterator = BasicIterator(batch_size=opt.batch_size)
    val_iterator.index_with(vocab)

    predictor = Predictor(iterator=val_iterator,
                          max_decoding_step=opt.max_step,
                          vocab=vocab,
                          reader=reader,
                          data_path=test_path,
                          log_dir=save_dir,
                          map_path=ner_path,
                          cuda_device=opt.gpu)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      #learning_rate_scheduler=learning_rate_scheduler,
                      learning_rate_decay=opt.lr_decay,
                      ema_decay=opt.ema_decay,
                      predictor=predictor, 
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=valid_dataset,
                      validation_metric='+bleu',
                      cuda_device=opt.gpu,
                      num_epochs=opt.epoch,
                      serialization_dir=save_dir,
                      num_serialized_models_to_keep=5,
                      #model_save_interval=60,
                      #summary_interval=500,
                      should_log_parameter_statistics=False,
                      grad_norm=10)

    trainer.train()


def evaluate():
    reader = PWKPReader()
    vocab = Vocabulary.from_files(vocab_dir)
    iterator = BasicIterator(batch_size=opt.batch_size)
    iterator.index_with(vocab)

    model = Seq2Seq(emb_size=opt.emb_size,
                    hidden_size=opt.hidden_size,
                    enc_layers = opt.enc_layers,
                    dec_layers = opt.dec_layers,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                    beam_size=opt.beam_size,
                    label_smoothing=opt.label_smoothing,
                    vocab=vocab)

    model = model.cuda(opt.gpu)
    model_state = torch.load(opt.restore, map_location=util.device_mapping(-1))
    model.load_state_dict(model_state)

    predictor = Predictor(iterator=iterator,
                          max_decoding_step=opt.max_step,
                          vocab=vocab,
                          reader=reader,
                          data_path=test_path,
                          log_dir=save_dir,
                          map_path=ner_path,
                          cuda_device=opt.gpu)
    
    predictor.evaluate(model)



if __name__ == '__main__':
    if opt.mode == 'train':
        train()
    else:
        evaluate()
