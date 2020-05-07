# Contextualized Sparse Representations (Sparc)
This repository provides author's implementation of [Contextualized Sparse Representation for Real-Time Open-Domain Question Answering](https://arxiv.org/abs/1911.02896). You can train and evaluate DenSPI+Sparc described in our paper and make your own Sparc vector.

## Environment
Please install the Conda environment as follows:
```bash
$ conda env create -f environment.yml
$ conda activate sparc
```
Note that this repository is mostly based on [DenSPI](https://github.com/uwnlp/denspi) and [DrQA](https://github.com/facebookresearch/DrQA).

## Resources
We use [SQuAD v1.1](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) for training DenSPI+Sparc. Please download them in `$DATA_DIR`.
```bash
$ mkdir $DATA_DIR
$ wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json -O $DATA_DIR/train-v1.1.json
$ wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json -O $DATA_DIR/dev-v1.1.json
```

DenSPI is based on BERT. Please download pre-trained weights of BERT under `$BERT_DIR`.
```bash
$ mkdir $BERT_DIR
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin -O $BERT_DIR/pytorch_model_base_uncased.bin
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json -O $BERT_DIR/bert_config_base_uncased.json
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin -O $BERT_DIR/pytorch_model_large_uncased.bin
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json -O $BERT_DIR/bert_config_large_uncased.json
# Vocabulary is the same for BERT-base and BERT-large.
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O $BERT_DIR/vocab.txt
```

## Model
To train DenSPI+Sparc, use `train.py`. Trained models will be saved in `$OUT_DIR1`.
```bash
$ mkdir $OUT_DIR1
# Train with BERT-base
$ python train.py --data_dir $DATA_DIR --metadata_dir $BERT_DIR --output_dir $OUT_DIR1 --bert_model_option 'base_uncased' --train_file train-v1.1.json --predict_file dev-v1.1.json --do_train --do_predict --do_eval
# Train with BERT-large (use smaller train_batch_size for 12GB GPUs)
$ python train.py --data_dir $DATA_DIR --metadata_dir $BERT_DIR --output_dir $OUT_DIR1 --bert_model_option 'large_uncased' --parallel --train_file train-v1.1.json --predict_file dev-v1.1.json --do_train --do_predict --do_eval --train_batch_size 6
```

The result will look like (in case of BERT-base):
```bash
(...)
04/28/2020 06:32:59 - INFO - post -   num vecs=45059736, num_words=1783576, nvpw=25.2637
04/28/2020 06:33:01 - INFO - __main__ -   [Validation] loss: 8.700, b'{"exact_match": 75.10879848628193, "f1": 83.42143097917004}\n'
```

To use DenSPI+Sparc in an open-domain setting, you have to additionally train it with negative samples. In case of DenSPI+Sparc with BERT-base (same for BERT-large except `--bert_model_option` and `--parallel` arguments), commands for training on negative samples are:
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

We also provide a pretrained DenSPI+Sparc as follows:
* DenSPI+Sparc pre-trained on SQuAD - [link](https://drive.google.com/file/d/1lObQ2lX8bWwJRzUuEqH6kpPdSTmS_Zxw/view?usp=sharing)


## Sparc Embedding
Given the pre-trained DenSPI+Sparc, you can get Sparc embedding with following commands. Example below assumes using our pre-trained weight ([`denspi_sparc.zip`](https://drive.google.com/file/d/1lObQ2lX8bWwJRzUuEqH6kpPdSTmS_Zxw/view?usp=sharing) unzipped in `denspi_sparc` folder). If you want to use your own model, please modify `MODEL_DIR` accordingly.

For any type of text you want to embed, put them in each line of `input_examples.txt`. Then run:
```bash
$ export DATA_DIR=.
$ export MODEL_DIR=denspi_sparc
$ python train.py --data_dir $DATA_DIR --metadata_dir $BERT_DIR --output_dir $OUT_DIR --predict_file input_examples.txt --parallel --bert_model_option 'large_uncased' --do_load --load_dir $MODEL_DIR --load_epoch 1 --do_embed --dump_file output.json
```

The result file `$OUT_DIR/output.json` will show Sparc embedding of the input text ([CLS] representation, sorted by scores). For instance:
```json
{
    "out": [
        {
            "text": "They defeated the Arizona Cardinals 49-15 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995.",
            "sparc": {
                "uni": {
                    "1995": {
                        "score": 1.6841894388198853,
                        "vocab": "2786"
                    },
                    "second": {
                        "score": 1.6321970224380493,
                        "vocab": "2117"
                    },
                    "49": {
                        "score": 1.6075607538223267,
                        "vocab": "4749"
                    },
                    "arizona": {
                        "score": 1.1734912395477295,
                        "vocab": "5334"
                    },
                },
                "bi": {
                    "arizona cardinals": {
                        "score": 1.3190401792526245,
                        "vocab": "5334, 9310"
                    },
                    "nfc championship": {
                        "score": 1.1005975008010864,
                        "vocab": "22309, 2528"
                    },
                    "49 -": {
                        "score": 1.0863999128341675,
                        "vocab": "4749, 1011"
                    },
                    "the arizona": {
                        "score": 0.9722453951835632,
                        "vocab": "1996, 5334"
                    },
                }
            }
        }
    ]
}
```
Note that each text is segmented by the BERT tokenizer (`"vocab"` denotes the BERT vocab index).

To see how Sparc changes for each phrase, set `start_index` in [here](https://github.com/jhyuklee/sparc/blob/750bf1a2b79f0e074edb77ef535c7e2861ffd8fd/post.py#L371) to the target token position. For instance, setting `start_index = 17` to embed Sparc of `415,000` of the following text gives you (some n-grams are omitted):

```json
            "text": "Between 1991 and 2000, the total area of forest lost in the Amazon rose from 415,000 to 587,000 square kilometres.",
            "sparc": {
                "uni": {
                    "1991": {
                        "score": 1.182684063911438,
                        "vocab": "2889"
                    },
                    "2000": {
                        "score": 0.41507360339164734,
                        "vocab": "2456"
                    },
```
whereas setting `start_index = 21` to embed Sparc of `587,000` gives you:
```json
...
            "text": "Between 1991 and 2000, the total area of forest lost in the Amazon rose from 415,000 to 587,000 square kilometres.",
            "sparc": {
                "uni": {
                    "2000": {
                        "score": 1.1923936605453491,
                        "vocab": "2456"
                    },
                    "1991": {
                        "score": 0.7090237140655518,
                        "vocab": "2889"
                    },
```

## Phrase Index
For now, please see [the original DenSPI repository](https://github.com/uwnlp/denspi) or [the recent application of DenSPI in COVID-19 domain](https://github.com/dmis-lab/covidAsk) for building phrase index using DenSPI+Sparc.
The main changes in phrase indexing is in `post.py` and `mips_phrase.py` where Sparc is used for the open-domain QA inference (See [here](https://github.com/jhyuklee/sparc/blob/885729372706e227fa9c566ca51bd88de984710a/mips_phrase.py#L390-L410)).

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
For any issues regarding Sparc, please register a GitHub issue.
