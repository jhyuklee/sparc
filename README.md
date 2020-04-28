# Contextualize Sparse Representations
This repository provides author's implementation of [Contextualized Sparse Representation for Real-Time Open-Domain Question Answering](https://arxiv.org). You can train and evaluate DenSPI+CoSpR described in our paper and make your own phrase index for a demo.

## Environment
Please install the Conda environment as follows:
```bash
$ conda env create -f environment.yml
$ conda activate cospr
```
Note that this repository is mostly based on [DenSPI](https://github.com/uwnlp/denspi) and [DrQA](https://github.com/facebookresearch/DrQA).

## Resources
We use [SQuAD v1.1](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) for training DenSPI+CoSpR. Please download them under any `$DATA_DIR`.
```bash
$ mkdir $DATA_DIR
$ wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json -O $DATA_DIR/train-v1.1.json
$ wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json -O $DATA_DIR/dev-v1.1.json
```

DenSPI is also based on BERT. Please download pre-trained weights of BERT under `$BERT_DIR`.
```bash
$ mkdir $BERT_DIR
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin -O $BERT_DIR/pytorch_model_base_uncased.bin
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json -O $BERT_DIR/bert_config_base_uncased.json
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin -O $BERT_DIR/pytorch_model_large_uncased.bin
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json -O $BERT_DIR/bert_config_large_uncased.json
# Vocabulary is the same for BERT-base and BERT-large.
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O $BERT_DIR/vocab.txt $ 
```

## Model
To train DenSPI+CoSpR, use `train.py`. Trained models will be saved in `$OUT_DIR1`.
```bash
$ mkdir $OUT_DIR1
# Train with BERT-base
$ python train.py --data_dir $DATA_DIR --metadata_dir $BERT_DIR --output_dir $OUT_DIR1 --bert_model_option 'base_uncased' --train_file train-v1.1.json --predict_file dev-v1.1.json --do_train --do_predict --do_eval
# Train with BERT-large (use smaller train_batch_size for 12GB GPUs)
$ python train.py --data_dir $DATA_DIR --metadata_dir $BERT_DIR --output_dir $OUT_DIR1 --bert_model_option 'large_uncased' --parallel --train_file train-v1.1.json --predict_file dev-v1.1.json --do_train --do_predict --do_eval --train_batch_size 6
```

The result will look like:
```bash
(...)
04/28/2020 06:32:59 - INFO - post -   num vecs=45059736, num_words=1783576, nvpw=25.2637
04/28/2020 06:33:01 - INFO - __main__ -   [Validation] loss: 8.700, b'{"exact_match": 75.10879848628193, "f1": 83.42143097917004}\n'
```

To use DenSPI+CoSpR in an open-domain setting, you have to additionally train it with negative samples. In case of DenSPI+CoSpR with BERT-base (similar for BERT-large except `--bert_model_option` and `--parallel` arguments), commands for training on negative samples are:
```bash
$ mkdir $OUT_DIR2
$ python train.py --data_dir $DATA_DIR --metadata_dir $BERT_DIR --output_dir $OUT_DIR --bert_model_option 'base_uncased' --train_file train-v1.1.json --predict_file dev-v1.1.json --do_train_neg --do_predict --do_eval --do_load --load_dir $OUT_DIR1 --load_epoch 3
```

Finally, train the phrase classifer as:
```bash
$ mkdir $OUT_DIR3
# Train only 1 epoch for phrase classifier
$ python train.py --data_dir $DATA_DIR --metadata_dir $BERT_DIR --output_dir $OUT_DIR --bert_model_option 'base_uncased' --train_file train-v1.1.json --predict_file dev-v1.1.json --num_train_epochs 1 --do_train_filter --do_predict --do_eval --do_load --load_dir $OUT_DIR2 --load_epoch 3
```

We also provide a pretrained DenSPI+CoSpR as follows:
* DenSPI+CoSpR pre-trained on SQuAD - [link](https://drive.google.com/open?id=1ft6_EAU1XtcBeCLmwkGXhemewppOs_SO)


## Phrase Dump
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
