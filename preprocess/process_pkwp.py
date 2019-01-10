import sys
sys.path.append('../')
import re
import json
import os
from typing import List, Dict, Iterator
from utils import jsonl
import torchfile

pwkp_data_path = os.path.expanduser('/data/data-simplification/wikismall/')

def transform(path: str) -> List[Dict]:
    source_datas = open(path+'.src', 'r').read().strip().split('\n')
    target_datas = open(path+'.dst', 'r').read().strip().split('\n')
    datas = []

    for s, t in zip(source_datas, target_datas):
        datas.append({'source': s.lower(), 'target': t.lower()})
    
    return datas

def get_aner_map(path: str) -> List:
    datas = torchfile.load(path, utf8_decode_strings=True)

    return datas


if __name__ == '__main__':
    train_datas = transform(os.path.join(pwkp_data_path, 'PWKP_108016.tag.80.aner.train'))
    test_datas = transform(os.path.join(pwkp_data_path, 'PWKP_108016.tag.80.aner.test'))
    valid_datas = transform(os.path.join(pwkp_data_path, 'PWKP_108016.tag.80.aner.valid'))
    aner_datas = get_aner_map(os.path.join(pwkp_data_path, 'PWKP_108016.tag.80.aner.map.t7'))

    jsonl.dumps(train_datas, open(os.path.join(pwkp_data_path, 'train.jsonl'), 'w'))
    jsonl.dumps(test_datas, open(os.path.join(pwkp_data_path, 'test.jsonl'), 'w'))
    jsonl.dumps(valid_datas, open(os.path.join(pwkp_data_path, 'dev.jsonl'), 'w'))
    json.dump(aner_datas, open(os.path.join(pwkp_data_path, 'aner.json'), 'w'))