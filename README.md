# Word Embedding Attention Network
Code for "Word Embedding Attention Network: Generating Words by Querying Distributed Word Representations for Paraphrase Generation"
## Requirements
* Ubuntu 16.04
* Python 3.5
* Pytorch 0.2.0
* [ROUGE](http://research.microsoft.com/~cyl/download/ROUGE-1.5.5.tgz)
## Data
* [LCSTS](http://icrc.hitsz.edu.cn/Article/show/139.html)
* [PWKP](https://github.com/XingxingZhang/dress)
* [EWSEW](https://github.com/senisioi/NeuralTextSimplification)
## Run
```bash
python3 preprocess.py -train_src TRAIN_SRC_DATA -train_tgt TRAIN_TGT_DATA
		      -test_src TEST_SRC_DATA -test_tgt TEST_TGT_DATA
		      -valid_src VALID_SRC_DATA -valid_tgt VALID_TGT_DATA
		      -save_data data/lcsts/lcsts.low.share.train.pt
		      -lower -share
```
```bash
python3 train.py -gpus 0 -score general -config lcsts.yaml -log wean
```
```bash
python3 predict.py -gpus 0 -score general -config lcsts.yaml -unk -restore data/lcsts/wean/best_rouge_checkpoint.pt
```
## Cite
To use this code, please cite the following paper:<br><br>
Shuming Ma, Xu Sun, Wei Li, Sujian Li, Wenjie Li, and Xuancheng Ren. 
Word Embedding Attention Network: Generating Words by Querying Distributed Word Representations for Paraphrase Generation. In proceedings of NAACL-HLT 2018.

bibtext:
```
@inproceedings{wean,
  author    = {Shuming Ma and Xu Sun and Wei Li and Sujian Li and Wenjie Li and Xuancheng Ren},
  title     = {Word Embedding Attention Network: Generating Words by Querying Distributed Word 
	       Representations for Paraphrase Generation},
  booktitle = {{NAACL} {HLT} 2018, The 2018 Conference of the North American Chapter
	       of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2018}
}
```