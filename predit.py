import torch
import torch.nn as nn
import torch.utils.data
import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim

import os
import argparse
import time
import math
import collections
import codecs

#config
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-config', default='default.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', action='store_true',
                    help="load pretrain embedding")
parser.add_argument('-limit', type=int, default=0,
                    help="data limit")
parser.add_argument('-log', default='', type=str,
                    help="log directory")
parser.add_argument('-unk', action='store_true',
                    help="replace unk")

opt = parser.parse_args()
config = utils.read_config(opt.config)

#checkpoint
if opt.restore:
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)
    config = checkpoints['config']

#cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus)>0
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    #cudnn.benchmark = True

#data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)
print('loading time cost: %.3f' % (time.time()-start_time))

testset = datas['test']
src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']
config.src_vocab = src_vocab.size()
config.tgt_vocab = tgt_vocab.size()

testloader = dataloader.get_loader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0)

if opt.pretrain:
    pretrain_embed = torch.load(config.emb_file)
else:
    pretrain_embed = None
#model
print('building model...\n')
if opt.model == 'seq2seq':
    model = models.seq2seq(config, src_vocab.size(), tgt_vocab.size(), use_cuda,
                           pretrain=pretrain_embed, score_fn=opt.score)
else:
    raise ValueError('Model not found!')


if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

#optimizer
if opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay,start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())

param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

#log
if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + str(int(time.time() * 1000)) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'log.txt')
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")

logging('total number of parameters: %d\n\n' % param_count)
logging('score function is %s\n\n' % opt.score)

#checkpoint
if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0
total_loss, start_time = 0, time.time()
report_total, report_correct = 0, 0
report_vocab, report_tot_vocab = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))


#evaluate
def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    for raw_src, src, src_len, raw_tgt, tgt, tgt_len in testloader:
        if len(opt.gpus) > 1:
            samples, alignment = model.module.sample(src, src_len)
        else:
            samples, alignment = model.beam_sample(src, src_len, beam_size=config.beam_size)

        candidate += [tgt_vocab.convertToLabels(s, dict.EOS) for s in samples]
        source += raw_src
        reference += raw_tgt
        alignments += [align for align in alignment]

    if opt.unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands

    score = {}

    if 'bleu' in config.metric:
        result = utils.eval_bleu(reference, candidate, log_path, config)
        score['bleu'] = float(result.split()[2][:-1])
        logging(result)

    if 'rouge' in config.metric:
        result = utils.eval_rouge(reference, candidate, log_path)
        try:
            score['rouge'] = result['F_measure'][0]
            logging("F_measure: %s Recall: %s Precision: %s\n"
                    % (str(result['F_measure']), str(result['recall']), str(result['precision'])))
        except:
            logging("Failed to compute rouge score.\n")
            score['rouge'] = 0.0


    return score



if __name__ == '__main__':
    eval(0)