# Contextualize Sparse Representations
This repository provides author's implementation of CoSpR (Contextualized Sparse Representation) for Real-Time Open-Domain Question Answering. You can train DenSPI+CoSpR and host a demo on your machine. See our [paper](http://arxiv.org) for more details.

## Environment
You can install the conda environment as follows:
```bash
$ conda env create -f environment.yml
$ conda activate cospr
```
Note that this repository is mostly based on [DenSPI](https://github.com/uwnlp/denspi) and [DrQA](https://github.com/facebookresearch/DrQA).

## Data
We use [SQuAD](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) for training DenSPI+CoSpR. Please download them and place them under any `$DATA_DIR`.

## Model
To train the model, 
We also provide a pretrained DenSPI+CoSpR as follows:
* DenSPI+CoSpR pre-trained on SQuAD - [link](https://drive.google.com/open?id=1ft6_EAU1XtcBeCLmwkGXhemewppOs_SO)

`models/denspi` is more suitable for long, formal questions (e.g., Is there concrete evidence for the presence of asymptomatic transmissions?) and `models/denspi-nq` is good at short questions (e.g., covid-19 origin).

## Phrase Dump
We use the 2020-04-10 CORD-19 dataset for making the phrase dumps. We provide two phrase dumps obtained from the two models above.
* 2020-04-10 with DenSPI (SQuAD) - `dumps/denspi_2020-04-10`
* 2020-04-10 with DenSPI (SQuAD + NQ) - `dumps/denspi-nq_2020-04-10`

To make your own phrase dumps with different articles, run `create_dump.sh`. If you are going to use one of the provided phrase dumps above, you can skip this part and go to the Hosting section. Make sure that the paths for pre-trained DenSPI and pre-processed datasets are pointing the right directories.
```bash
$ ./create_dump.sh
```
This will create a new phrase dump under `dumps_new/$MODEL_$DATA`. Note that it will take approximately 1 hour when using `data/2020-04-10`. See log files in `logs/` to check if dumping is done. After the dumping, you need to run `create_index.sh` to make tfidf vectors of documents and paragraphs, and MIPS for phrase vectors.
```bash
$ ./create_index.sh
```
Before running, please change the directories in `create_index.sh` accordingly.

## Hosting
To serve your own covidAsk, use `serve.sh` script.
```bash
$ ./serve.sh
```
This will host a new server in localhost with the specified port (default `$PORT`: 9030). You will also need to serve query encoder (default `$Q_PORT`: 9010) and the metadata (default `$D_PORT`: 9020) at separate ports. Note that the model used for query encoding should be the same as the model that created the phrase dump. If you want to change the phrase dump to what you have created, change `$DUMP_DIR` to the new phrase dump (e.g., `DUMP_DIR=dumps_new/denspi_2020-04-10`) and `--doc_ranker_name` used in `d_serve` to `$DATA-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz`. We also use biomedical entity search engine, [BEST](https://best.korea.ac.kr), to provide further information regarding the entities in the query.

## Reference
```
@inproceedings{lee2020contextualized,
  title={Contextualized Sparse Representations for Real-Time Open-Domain Question Answering},
  author={Lee, Jinhyuk and Seo, Minjoon and Hajishirzi, Hannaneh and Kang, Jaewoo},
  booktitle={ACL},
  year={2020}
}
```

## Contact
For any issues regarding CoSpR, please register a GitHub issue.
