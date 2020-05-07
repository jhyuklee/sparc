# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import argparse
import collections
import logging
import json
import os
import random
import subprocess
from tqdm import tqdm as tqdm_
from time import time

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam

import tokenization
import utils
from modeling import BertConfig, DenSPI
from optimization import BERTAdam
from post import write_predictions, write_hdf5, get_question_results, \
    convert_question_features_to_dataloader, write_question_results, write_embed
from pre import convert_examples_to_features, read_squad_examples, convert_documents_to_features, \
    convert_questions_to_features, SquadExample, inject_noise_to_neg_features_list, sample_similar_questions, \
    compute_tfidf, read_text_examples


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "all_logits", "filter_start_logits", "filter_end_logits", "loss"]
)
ContextResult = collections.namedtuple(
    "ContextResult",
    ['unique_id', 'start', 'end', 'span_logits', 'filter_start_logits', 'filter_end_logits', 'start_sp', 'end_sp']
)


def tqdm(*args, mininterval=5.0, **kwargs):
    return tqdm_(*args, mininterval=mininterval, **kwargs)


def main():
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument("--train_file", default='train-v1.1.json', type=str, help="json for training.")
    parser.add_argument("--predict_file", default='dev-v1.1.json', type=str, help="json for prediction.")
    parser.add_argument('--has_answers', default=True, action='store_true', help="if predict file has position answer")

    # Metadata paths
    parser.add_argument('--metadata_dir', default='metadata/', type=str, help="Dir for pre-trained models.")
    parser.add_argument("--vocab_file", default='vocab.txt', type=str, help="Vocabulary file of pre-trained model.")
    parser.add_argument("--bert_model_option", default='large_uncased', type=str,
                        help="model architecture option. [large_uncased] or [base_uncased].")
    parser.add_argument("--bert_config_file", default='bert_config.json', type=str,
                        help="The config json file corresponding to the pre-trained BERT model.")
    parser.add_argument("--init_checkpoint", default='pytorch_model.bin', type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")

    # Output and load paths
    parser.add_argument("--output_dir", default='out/', type=str, help="storing models and predictions")
    parser.add_argument("--dump_dir", default='test/', type=str)
    parser.add_argument("--dump_file", default='phrase.hdf5', type=str, help="dump phrases of file.")
    parser.add_argument("--train_question_emb_file", default='train_question.hdf5', type=str, help="Used for neg train.")
    parser.add_argument('--load_dir', default='out/', type=str, help="Dir for checkpoints of models to load.")
    parser.add_argument('--load_epoch', type=str, default='1', help="Epoch of model to load.")

    # Do's
    parser.add_argument("--do_load", default=False, action='store_true', help='Do load. If eval, do load automatically')
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train_neg", default=False, action='store_true', help="Whether to run neg training.")
    parser.add_argument("--do_train_filter", default=False, action='store_true', help='Train filter or not.')
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--do_dump', default=False, action='store_true')
    parser.add_argument('--do_embed', default=False, action='store_true')

    # Model options: if you change these, you need to train again
    parser.add_argument("--do_case", default=False, action='store_true', help="Whether to keep upper casing")
    parser.add_argument("--use_sparse", default=True, action='store_true')
    parser.add_argument("--sparse_ngrams", default='1,2', type=str)
    parser.add_argument("--neg_with_tfidf", default=False, action='store_true')
    parser.add_argument("--skip_no_answer", default=False, action='store_true')
    parser.add_argument('--freeze_word_emb', default=True, action='store_true')

    # GPU and memory related options
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="Questions longer than this will be truncated to this length.")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=64, type=int, help="Total batch size for predictions.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    # Training options: only effective during training
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")

    # Prediction options: only effective during prediction
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    # Index Options
    parser.add_argument('--dtype', default='float32', type=str)
    parser.add_argument('--filter_threshold', default=-1e9, type=float)
    parser.add_argument('--dense_offset', default=-2, type=float) # Original
    parser.add_argument('--dense_scale', default=20, type=float)
    parser.add_argument('--sparse_offset', default=1.6, type=float)
    parser.add_argument('--sparse_scale', default=80, type=float)

    # Others
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--seed', type=int, default=45,
                        help="random seed for initialization")
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--draft_num_examples', type=int, default=12)

    args = parser.parse_args()

    # Filesystem routines
    class Processor(object):
        def __init__(self, save_path, load_path):
            self._save = None
            self._load = None
            self._save_path = save_path
            self._load_path = load_path

        def bind(self, save, load):
            self._save = save
            self._load = load

        def save(self, checkpoint=None, save_fn=None, **kwargs):
            path = os.path.join(self._save_path, str(checkpoint))
            if save_fn is None:
                self._save(path, **kwargs)
            else:
                save_fn(path, **kwargs)

        def load(self, checkpoint, load_fn=None, session=None, **kwargs):
            assert self._load_path == session
            path = os.path.join(self._load_path, str(checkpoint), 'model.pt')
            if load_fn is None:
                self._load(path, **kwargs)
            else:
                load_fn(path, **kwargs)

    # Save/Load function binding
    def bind_model(processor, model, optimizer=None):
        def save(filename, save_model=True, saver=None, **kwargs):
            if not os.path.exists(filename):
                os.makedirs(filename)
            if save_model:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                model_path = os.path.join(filename, 'model.pt')
                dummy_path = os.path.join(filename, 'dummy')
                torch.save(state, model_path)
                with open(dummy_path, 'w') as fp:
                    json.dump([], fp)
                logger.info('Model saved at %s' % model_path)
            if saver is not None:
                saver(filename)
        def load(filename, load_model=True, loader=None, **kwargs):
            if load_model:
                # logger.info('%s: %s' % (filename, os.listdir(filename)))
                model_path = os.path.join(filename, 'model.pt')
                if not os.path.exists(model_path):  # for compatibility
                    model_path = filename
                state = torch.load(model_path, map_location='cpu')
                sample_weight = list(state['model'].keys())[10]
                logger.info(f'Loaded weight has {state["model"][sample_weight][0]} in {sample_weight}')
                logger.info(f'Before loading, model has {model.state_dict()[sample_weight][0]} in {sample_weight}')
                try:
                    model.load_state_dict(state['model'])
                    if optimizer is not None:
                        optimizer.load_state_dict(state['optimizer'])
                    logger.info('load okay')
                except:
                    # Backward compatibility
                    model.load_state_dict(state['model'], strict=False)
                    utils.check_diff(model.state_dict(), state['model'])
                logger.info(f'After loading, model has {model.state_dict()[sample_weight][0]} in {sample_weight}')
                logger.info('Model loaded from %s' % model_path)
            if loader is not None:
                loader(filename)
        processor.bind(save=save, load=load)

    processor = Processor(args.output_dir, args.load_dir)
    if not(args.do_train or args.do_train_neg):
        if args.do_load is False:
            logger.info("Setting do_load to true for prediction")
            args.do_load = True

    # Configure file paths
    args.train_file = os.path.join(args.data_dir, args.train_file)
    args.predict_file = os.path.join(args.data_dir, args.predict_file)

    # Configure metadata paths
    args.vocab_file = os.path.join(args.metadata_dir, args.vocab_file)
    args.bert_config_file = os.path.join(
        args.metadata_dir, args.bert_config_file.replace(".json", "") + "_" + args.bert_model_option + ".json"
    )
    args.init_checkpoint = os.path.join(
        args.metadata_dir, args.init_checkpoint.replace(".bin", "") + "_" + args.bert_model_option + ".bin"
    )

    # Output paths
    args.dump_file = os.path.join(args.dump_dir, args.dump_file)
    args.train_question_emb_file = os.path.join(args.output_dir, args.train_question_emb_file)

    # CUDA Check
    logger.info('cuda availability: {}'.format(torch.cuda.is_available()))
    if not torch.cuda.is_available() and (args.do_train or args.do_train_neg or args.do_train_filter):
        logger.info('We do not support training with CPUs.')
        exit()

    # Multi-GPU stuff
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.info("Overwriting outputs in %s"% args.output_dir)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.dump_dir) and os.listdir(args.dump_dir):
        logger.info("Overwriting dump in %s"% args.dump_dir)
    else:
        os.makedirs(args.dump_dir, exist_ok=True)

    # Get model and tokenizer
    model = DenSPI(bert_config,
        sparse_ngrams=args.sparse_ngrams.split(','),
        use_sparse=args.use_sparse,
        neg_with_tfidf=args.neg_with_tfidf,
        do_train_filter=args.do_train_filter,
        min_word_id=999 if not args.do_case else 106
    )
    logger.info('Number of model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=not args.do_case)

    # Initialize BERT if not loading and has init_checkpoint
    if not args.do_load and args.init_checkpoint is not None:
        if args.draft:
            logger.info('[Draft] Randomly initialized model')
        else:
            state_dict = torch.load(args.init_checkpoint, map_location='cpu')
            if next(iter(state_dict)).startswith('bert.'):
                state_dict = {key[len('bert.'):]: val for key, val in state_dict.items()}
                state_dict = {key: val for key, val in state_dict.items() if key in model.bert.state_dict()}
            utils.check_diff(model.bert.state_dict(), state_dict)
            model.bert.load_state_dict(state_dict)
            logger.info('Model initialized from the pre-trained BERT weight!')

    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif args.parallel or n_gpu > 1:
        model = torch.nn.DataParallel(model)
        logger.info("Data parallel!")

    if args.do_load:
        bind_model(processor, model)
        processor.load(args.load_epoch, session=args.load_dir)

    def is_freeze_param(name):
        if args.freeze_word_emb:
            if name.endswith("bert.embeddings.word_embeddings.weight"):
                logger.info(f'freezeing {name}')
                return False
        return True

    # Prediction file loader
    def get_eval_data(args):
        logger.info(f'Does prediction file have answer positions?: {args.has_answers}')
        eval_examples = read_squad_examples(
            input_file=args.predict_file,
            return_answers=args.has_answers,
            draft=args.draft,
            draft_num_examples=args.draft_num_examples)
        eval_features, query_eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            return_answers=args.has_answers,
            skip_no_answer=False,
            msg="Converting eval examples")

        # Convert to features
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_input_ids_ = torch.tensor([f.input_ids for f in query_eval_features], dtype=torch.long)
        all_input_mask_ = torch.tensor([f.input_mask for f in query_eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        if args.has_answers:
            all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)

        if args.fp16:
            (all_input_ids, all_input_mask,
             all_example_index) = tuple(t.half() for t in (all_input_ids, all_input_mask, all_example_index))
            (all_input_ids_, all_input_mask_) = tuple(t.half() for t in (all_input_ids_, all_input_mask_))

        if args.has_answers:
            eval_data = TensorDataset(all_input_ids, all_input_mask,
                                      all_input_ids_, all_input_mask_,
                                      all_start_positions, all_end_positions,
                                      all_example_index)
        else:
            eval_data = TensorDataset(all_input_ids, all_input_mask,
                                      all_input_ids_, all_input_mask_,
                                      all_input_ids_, all_input_mask_, # Just copy for shape consistency
                                      all_example_index)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        return eval_dataloader, eval_examples, eval_features

    # Prediction wrapper
    def predict_with_args(model, eval_data, args):
        if not args.do_predict:
            logger.info('do_predict turned off')
            return

        eval_dataloader, eval_examples, eval_features = eval_data
        model.eval()

        def get_results():
            for (input_ids, input_mask, input_ids_, input_mask_, start_positions, end_positions,
                 example_indices) in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                input_ids_ = input_ids_.to(device)
                input_mask_ = input_mask_.to(device)
                if args.has_answers:
                    start_positions = start_positions.to(device)
                    end_positions = end_positions.to(device)
                with torch.no_grad():
                    if args.has_answers:
                        batch_loss, _, batch_all_logits, batch_filter_start_logits, batch_filter_end_logits = model(
                            input_ids, input_mask, input_ids_, input_mask_, start_positions, end_positions)
                    else:
                        batch_all_logits, batch_filter_start_logits, batch_filter_end_logits = model(
                            input_ids, input_mask, input_ids_, input_mask_)

                for i, example_index in enumerate(example_indices):
                    all_logits = batch_all_logits[i].detach().triu().view(-1).softmax(dim=-1)
                    all_logits = all_logits.view(*batch_all_logits[i].shape)
                    all_logits = all_logits.cpu().numpy()
                    filter_start_logits = batch_filter_start_logits[i].detach().cpu().numpy()
                    filter_end_logits = batch_filter_end_logits[i].detach().cpu().numpy()
                    loss = 0
                    if args.has_answers:
                        loss = batch_loss.mean().item() # we are approximating the loss by batch-wise
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    yield RawResult(unique_id=unique_id,
                                    all_logits=all_logits,
                                    filter_start_logits=filter_start_logits,
                                    filter_end_logits=filter_end_logits,
                                    loss=loss)

        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_score_file = os.path.join(args.output_dir, "scores.json")
        val_loss = write_predictions(eval_examples, eval_features, get_results(),
                              args.max_answer_length,
                              not args.do_case, output_prediction_file,
                              output_score_file, args.verbose_logging,
                              args.filter_threshold)
        
        # For SQuAD-style evaluation
        if args.do_eval:
            command = "python evaluate-v1.1.py %s %s" % (args.predict_file, output_prediction_file)
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            metrics = output
            logger.info(f"[Validation] loss: {val_loss:.3f}, {output}")

    # Ready for the eval data
    eval_data = None
    if args.do_predict:
        eval_data = get_eval_data(args)

    # Train
    if args.do_train:
        print()
        train_examples = read_squad_examples(
            input_file=args.train_file, return_answers=True, draft=args.draft,
            draft_num_examples=args.draft_num_examples)
        train_features, train_features_ = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            return_answers=True,
            skip_no_answer=False,
            verbose=True,
            msg="Converting train examples")

        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        global_step = 0

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if (n not in no_decay) and is_freeze_param(n)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if (n in no_decay) and is_freeze_param(n)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = BERTAdam(optimizer_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)
        bind_model(processor, model, optimizer)

        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        all_input_ids_ = torch.tensor([f.input_ids for f in train_features_], dtype=torch.long)
        all_input_mask_ = torch.tensor([f.input_mask for f in train_features_], dtype=torch.long)

        if args.fp16:
            (all_input_ids, all_input_mask,
             all_start_positions,
             all_end_positions) = tuple(t.half() for t in (all_input_ids, all_input_mask,
                                                           all_start_positions, all_end_positions))
            all_input_ids_, all_input_mask_ = tuple(t.half() for t in (all_input_ids_, all_input_mask_))

        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_input_ids_, all_input_mask_,
                                   all_start_positions, all_end_positions)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Basic Training
        for epoch in range(int(args.num_train_epochs)):
            print()
            model.train()
            total_loss = 0.0
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for step, batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                (input_ids, input_mask,
                 input_ids_, input_mask_,
                 start_positions, end_positions) = batch
                loss, _, _, _, _ = model(input_ids, input_mask,
                                input_ids_, input_mask_,
                                start_positions, end_positions)
                pbar.set_description("[Epoch %d] Train loss: %.3f" % (epoch+1, loss.mean().item()))
                total_loss += loss.mean().item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()  # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

            logger.info("[Epoch %d] Average train loss: %.2f"% (epoch+1, total_loss / len(train_dataloader)))
            predict_with_args(model, eval_data, args)
            processor.save(epoch + 1)

    # Train with negative samples
    if args.do_train_neg:

        # Embed question for train neg
        question_examples = read_squad_examples(
            question_only=True,
            input_file=args.train_file, return_answers=False, draft=args.draft,
            draft_num_examples=args.draft_num_examples)
        query_eval_features = convert_questions_to_features(
            examples=question_examples,
            tokenizer=tokenizer,
            max_query_length=args.max_query_length)
        question_dataloader = convert_question_features_to_dataloader(query_eval_features, args.fp16, args.local_rank,
                                                                      args.predict_batch_size)

        model.eval()
        logger.info("Start embedding")
        question_results = get_question_results(question_examples, query_eval_features, question_dataloader, device,
                                                model)
        logger.info('Writing %s' % args.train_question_emb_file)
        write_question_results(question_results, query_eval_features, args.train_question_emb_file)

        train_examples = read_squad_examples(
            input_file=args.train_file, return_answers=True, draft=args.draft,
            draft_num_examples=args.draft_num_examples)

        global_step = 0
        train_features, train_features_ = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            return_answers=True,
            skip_no_answer=False,
            verbose=True)

        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if (n not in no_decay) and is_freeze_param(n)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if (n in no_decay) and is_freeze_param(n)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = BERTAdam(optimizer_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)
        bind_model(processor, model, optimizer)

        neg_train_features = sample_similar_questions(train_examples, train_features, args.train_question_emb_file,
                                                      cuda=not args.no_cuda)
        neg_train_features = inject_noise_to_neg_features_list(neg_train_features,
                                                               noise_prob=0.2,
                                                               clamp=True, clamp_prob=0.1,
                                                               replace=True, replace_prob=0.1, unk_prob=0.1,
                                                               shuffle=True, shuffle_prob=0.1,
                                                               vocab_size=len(tokenizer.vocab),
                                                               min_id=999 if not args.do_case else 106)

        if args.neg_with_tfidf:
            tfidf_features = compute_tfidf(train_examples, train_features, neg_train_features, train_features_,
                args.data_dir)
            pos_scores = torch.tensor([f.pos_score for f in tfidf_features], dtype=torch.float).squeeze()
            neg_scores = torch.tensor([f.neg_score for f in tfidf_features], dtype=torch.float).squeeze()
        else:
            pos_scores = torch.tensor([0.0 for _ in range(len(train_features))], dtype=torch.float)
            neg_scores = torch.tensor([0.0 for _ in range(len(train_features))], dtype=torch.float)

        logger.info("***** Running neg training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        all_input_ids_ = torch.tensor([f.input_ids for f in train_features_], dtype=torch.long)
        all_input_mask_ = torch.tensor([f.input_mask for f in train_features_], dtype=torch.long)

        all_neg_input_ids = torch.tensor([f.input_ids for f in neg_train_features], dtype=torch.long)
        all_neg_input_mask = torch.tensor([f.input_mask for f in neg_train_features], dtype=torch.long)

        if args.fp16:
            (all_input_ids, all_input_mask,
             all_start_positions,
             all_end_positions) = tuple(t.half() for t in (all_input_ids, all_input_mask,
                                                           all_start_positions, all_end_positions))
            all_input_ids_, all_input_mask_ = tuple(t.half() for t in (all_input_ids_, all_input_mask_))
            all_neg_input_ids, all_neg_input_mask = tuple(t.half() for t in (all_neg_input_ids, all_neg_input_mask))

        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_input_ids_, all_input_mask_,
                                   all_start_positions, all_end_positions,
                                   all_neg_input_ids, all_neg_input_mask,
                                   pos_scores, neg_scores)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Neg Training
        for epoch in range(int(args.num_train_epochs)):
            model.train()
            total_loss = 0.0
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for step, batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                (input_ids, input_mask,
                 input_ids_, input_mask_,
                 start_positions, end_positions,
                 neg_input_ids, neg_input_mask,
                 pos_score, neg_score) = batch
                loss, _, _, _, _ = model(input_ids, input_mask,
                                input_ids_, input_mask_,
                                start_positions, end_positions,
                                neg_input_ids, neg_input_mask,
                                pos_score, neg_score)
                pbar.set_description("[Epoch %d] Train loss: %.3f" % (epoch+1, loss.mean().item()))
                total_loss += loss.mean().item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()  # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

            logger.info("[Epoch %d] Average train neg loss: %.3f"% (epoch+1, total_loss / len(train_dataloader)))
            predict_with_args(model, eval_data, args)
            processor.save(epoch + 1)

    # Train Filter
    if args.do_train_filter:
        train_examples = read_squad_examples(
            input_file=args.train_file, return_answers=True, draft=args.draft,
            draft_num_examples=args.draft_num_examples)

        global_step = 0
        train_features, train_features_ = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            return_answers=True,
            skip_no_answer=True)

        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        if args.parallel or n_gpu > 1:
            optimizer = Adam(model.module.linear.parameters())
        else:
            optimizer = Adam(model.linear.parameters())

        bind_model(processor, model, optimizer)
        logger.info("***** Running filter training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

        all_input_ids_ = torch.tensor([f.input_ids for f in train_features_], dtype=torch.long)
        all_input_mask_ = torch.tensor([f.input_mask for f in train_features_], dtype=torch.long)

        if args.fp16:
            (all_input_ids, all_input_mask,
             all_start_positions,
             all_end_positions) = tuple(t.half() for t in (all_input_ids, all_input_mask,
                                                           all_start_positions, all_end_positions))
            all_input_ids_, all_input_mask_ = tuple(t.half() for t in (all_input_ids_, all_input_mask_))

        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_input_ids_, all_input_mask_,
                                   all_start_positions, all_end_positions)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Filter Training
        model.train()
        for epoch in range(int(args.num_train_epochs)):
            total_loss = 0.0
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for step, batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                (input_ids, input_mask,
                 input_ids_, input_mask_,
                 start_positions, end_positions) = batch
                _, loss, _, _, _= model(input_ids, input_mask,
                                input_ids_, input_mask_,
                                start_positions, end_positions)
                pbar.set_description("[Epoch %d] Train loss: %.3f" % (epoch+1, loss.mean().item()))
                total_loss += loss.mean().item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()  # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

            processor.save(epoch + 1)
            logger.info("Train filter loss: %.2f"% (total_loss / len(train_dataloader)))

    # Run prediction
    if args.do_predict:
        print()
        predict_with_args(model, eval_data, args)

    # Dump phrases
    if args.do_dump:
        if ':' not in args.predict_file:
            predict_files = [args.predict_file]
            offsets = [0]
        else:
            dirname = os.path.dirname(args.predict_file)
            basename = os.path.basename(args.predict_file)
            start, end = list(map(int, basename.split(':')))

            # skip files if possible
            if os.path.exists(args.dump_file):
                with h5py.File(args.dump_file, 'r') as f:
                    dids = list(map(int, f.keys()))
                start = int(max(dids) / 1000)
                logger.info('%s exists; starting from %d' % (args.dump_file, start))

            names = [str(i).zfill(4) for i in range(start, end)]
            predict_files = [os.path.join(dirname, name) for name in names]
            offsets = [int(each) * 1000 for each in names]

        for offset, predict_file in zip(offsets, predict_files):
            context_examples = read_squad_examples(
                context_only=True,
                input_file=predict_file, return_answers=False, draft=args.draft,
                draft_num_examples=args.draft_num_examples)

            for example in context_examples:
                example.doc_idx += offset

            context_features = convert_documents_to_features(
                examples=context_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride)

            logger.info("***** Running dumping on %s *****" % predict_file)
            logger.info("  Num orig examples = %d", len(context_examples))
            logger.info("  Num split examples = %d", len(context_features))
            logger.info("  Batch size = %d", args.predict_batch_size)

            all_input_ids = torch.tensor([f.input_ids for f in context_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in context_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            if args.fp16:
                all_input_ids, all_input_mask, all_example_index = tuple(
                    t.half() for t in (all_input_ids, all_input_mask, all_example_index))

            context_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)

            if args.local_rank == -1:
                context_sampler = SequentialSampler(context_data)
            else:
                context_sampler = DistributedSampler(context_data)
            context_dataloader = DataLoader(context_data, sampler=context_sampler,
                                            batch_size=args.predict_batch_size)

            model.eval()
            logger.info("Start dumping")

            def get_context_results():
                for (input_ids, input_mask, example_indices) in context_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    with torch.no_grad():
                        batch_start, batch_end, batch_span_logits, batch_filter_start, batch_filter_end, sp_s, sp_e = model(
                                                                                                    input_ids=input_ids,
                                                                                                    input_mask=input_mask)
                    for i, example_index in enumerate(example_indices):
                        start = batch_start[i].detach().cpu().numpy().astype(args.dtype)
                        end = batch_end[i].detach().cpu().numpy().astype(args.dtype)
                        sparse = None
                        if len(sp_s) > 0:
                            b_ssp = {ng: bb_ssp[i].detach().cpu().numpy().astype(args.dtype) for ng, bb_ssp in sp_s.items()}
                            b_esp = {ng: bb_esp[i].detach().cpu().numpy().astype(args.dtype) for ng, bb_esp in sp_e.items()}
                            # b_ssp = sp_s[i].detach().cpu().numpy().astype(args.dtype)
                            # b_esp = sp_e[i].detach().cpu().numpy().astype(args.dtype)
                        span_logits = batch_span_logits[i].detach().cpu().numpy().astype(args.dtype)
                        filter_start_logits = batch_filter_start[i].detach().cpu().numpy().astype(args.dtype)
                        filter_end_logits = batch_filter_end[i].detach().cpu().numpy().astype(args.dtype)
                        context_feature = context_features[example_index.item()]
                        unique_id = int(context_feature.unique_id)
                        yield ContextResult(unique_id=unique_id,
                                            start=start,
                                            end=end,
                                            span_logits=span_logits,
                                            filter_start_logits=filter_start_logits,
                                            filter_end_logits=filter_end_logits,
                                            start_sp=b_ssp,
                                            end_sp=b_esp)

            t0 = time()
            write_hdf5(context_examples, context_features, get_context_results(),
                       args.max_answer_length, not args.do_case, args.dump_file, args.filter_threshold,
                       args.verbose_logging,
                       dense_offset=args.dense_offset, dense_scale=args.dense_scale,
                       sparse_offset=args.sparse_offset, sparse_scale=args.sparse_scale,
                       use_sparse=args.use_sparse)
            logger.info('%s: %.1f mins' % (predict_file, (time() - t0) / 60))

    if args.do_embed:
        context_examples = read_text_examples(
            input_file=args.predict_file, draft=args.draft, draft_num_examples=args.draft_num_examples
        )

        context_features = convert_documents_to_features(
            examples=context_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride)

        logger.info("***** Embedding %s *****" % args.predict_file)
        logger.info("  Num orig examples = %d", len(context_examples))
        logger.info("  Num split examples = %d", len(context_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in context_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in context_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        context_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)

        if args.local_rank == -1:
            context_sampler = SequentialSampler(context_data)
        else:
            context_sampler = DistributedSampler(context_data)
        context_dataloader = DataLoader(context_data, sampler=context_sampler,
                                        batch_size=args.predict_batch_size)

        model.eval()
        logger.info("Start embedding")

        def get_context_results():
            for (input_ids, input_mask, example_indices) in context_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                with torch.no_grad():
                    batch_start, batch_end, batch_span_logits, batch_filter_start, batch_filter_end, sp_s, sp_e = model(
                                                                                                input_ids=input_ids,
                                                                                                input_mask=input_mask)
                for i, example_index in enumerate(example_indices):
                    start = batch_start[i].detach().cpu().numpy().astype(args.dtype)
                    end = batch_end[i].detach().cpu().numpy().astype(args.dtype)
                    sparse = None
                    if len(sp_s) > 0:
                        b_ssp = {ng: bb_ssp[i].detach().cpu().numpy().astype(args.dtype) for ng, bb_ssp in sp_s.items()}
                        b_esp = {ng: bb_esp[i].detach().cpu().numpy().astype(args.dtype) for ng, bb_esp in sp_e.items()}
                    span_logits = batch_span_logits[i].detach().cpu().numpy().astype(args.dtype)
                    filter_start_logits = batch_filter_start[i].detach().cpu().numpy().astype(args.dtype)
                    filter_end_logits = batch_filter_end[i].detach().cpu().numpy().astype(args.dtype)
                    context_feature = context_features[example_index.item()]
                    unique_id = int(context_feature.unique_id)
                    yield ContextResult(unique_id=unique_id,
                                        start=start,
                                        end=end,
                                        span_logits=span_logits,
                                        filter_start_logits=filter_start_logits,
                                        filter_end_logits=filter_end_logits,
                                        start_sp=b_ssp,
                                        end_sp=b_esp)

        t0 = time()
        write_embed(context_examples, context_features, get_context_results(),
                    args.max_answer_length, not args.do_case, args.dump_file)
        logger.info('%s: %.1f mins' % (args.predict_file, (time() - t0) / 60))


if __name__ == "__main__":
    main()
