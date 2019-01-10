from allennlp.common import Registrable
from allennlp.models import Model
from allennlp.data import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from typing import Iterator, List, Dict
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Set
from metrics import calc_bleu_score
import json

class Predictor(Registrable):

    def __init__(self,
                 iterator: DataIterator,
                 max_decoding_step: int,
                 vocab: Vocabulary,
                 reader: DatasetReader,
                 data_path: str,
                 log_dir: str,
                 map_path: str = None, 
                 cuda_device: Union[int, List] = -1) -> None:
    
        self.iterator = iterator
        self.max_decoding_step = max_decoding_step
        self.vocab = vocab
        self.reader = reader
        self.dataset = reader.read(data_path)
        self.data_path = data_path
        self.log_dir = log_dir
        self.cuda_device = cuda_device
        if map_path is not None:
            self.post_map = json.load(open(map_path))['test']
        else:
            self.post_map = None

    def post_processs(self, line: str, maps: Dict[str, str]) -> str:
        _words = []
        for token in line.strip().split(' '):
            if token.upper() in maps:
                _words.append(maps[token.upper()].lower())
            else:
                _words.append(token)

        return ' '.join(_words)
    
    def evaluate(self, model: Model):
        model.eval()

        val_generator = self.iterator(self.dataset,
                                      num_epochs=1,
                                      shuffle=False)

        num_validation_batches = self.iterator.get_num_batches(self.dataset)
        val_generator_tqdm = Tqdm.tqdm(val_generator, total=num_validation_batches)
        vocabulary = self.vocab.get_index_to_token_vocabulary('tokens')

        predictions, sources, references, alignments = [], [], [], []
        
        for data in self.reader.read_raw(self.data_path):
            sources.append(data['source'])
            references.append(data['target'])

        for batch in val_generator_tqdm:
            batch = util.move_to_device(batch, self.cuda_device)

            output_dict = model.predict(batch['src'], max_decoding_step=self.max_decoding_step)
            alignments += output_dict['alignments']

            for pred in output_dict['output_ids']:
                pred_sent = list(map(vocabulary.get, pred))
                if '@@EOS@@' in pred_sent:
                    pred_sent = pred_sent[:pred_sent.index('@@EOS@@')]
                pred_sent = ' '.join(pred_sent)
                predictions.append(pred_sent)
        
        for i in range(len(predictions)):
            source_sent = sources[i].split(' ')
            pred_sent = predictions[i].split(' ')
            for j in range(len(pred_sent)):
                if pred_sent[j] == '@@UNKNOWN@@' and alignments[i][j] < len(source_sent):
                    pred_sent[j] = source_sent[alignments[i][j]]
            predictions[i] = ' '.join(pred_sent)

        if self.post_map is not None:
            predictions = [self.post_processs(p, m) for p, m in zip(predictions, self.post_map)]
            references = [self.post_processs(r, m) for r, m in zip(references, self.post_map)]

        score = {}
        score['bleu'] = calc_bleu_score(predictions, references, self.log_dir)
        model.train()

        return score