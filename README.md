# Word Embedding Attention Network

Code for "Word Embedding Attention Network: Generating Words by Querying Distributed Word Representations for Paraphrase Generation" [[pdf]](https://arxiv.org/abs/1803.01465)

## Requirements

* Ubuntu 16.04
* Python 3.6
* Pytorch 1.0.0
* allennlp 0.7.2
* torchfile

## Data Preparation

- Step 1: Download the [PWKP dataset](https://github.com/XingxingZhang/dress) and put it in the folder *data/*.
- Step 2: Preprocess the dataset
```bash
cd preprocess/
python3 process_pkwp.py
```

## Run

- Step 1: Train a model
```bash
python3 run.py -gpu 0 -mode train -dir save_path
```
- Step 2: Restore and evaluate the model with the BLEU metric
```bash
python3 run.py -gpu 0 -mode evaluate -restore save_path/best.th
```

## Pretrained Model

The code is currently non-deterministic due to various GPU ops, so you are likely to end up with a slightly better or worse evaluation. We provide a [pretrained model](https://drive.google.com/open?id=1IJ6LM_YVJHSPcAfwCeRyOGraO9k3dkme) to reproduce the results reported in our paper.


## Cite
Hopefully the codes and the datasets are useful for the future research. If you use the above codes or datasets for your research, please kindly cite the following paper:

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