import collections
import json
import logging
import h5py
import os

import pandas as pd
import six
import random
import torch
import pickle as pkl
import numpy as np
import copy
import unicodedata
from tqdm import tqdm
from tfidf_doc_ranker import TfidfDocRanker

import tokenization
from post import _improve_answer_span, _check_is_max_context, get_final_text_

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
NO_ANS = -1


class SquadExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id=None,
                 question_text=None,
                 paragraph_text=None,
                 doc_words=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 title="",
                 doc_idx=0,
                 par_idx=0):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.doc_words = doc_words
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.title = title
        self.doc_idx = doc_idx
        self.par_idx = par_idx

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (tokenization.printable_text(self.question_text))
        s += ", paragraph_text: %s" % (tokenization.printable_text(self.paragraph_text))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s



class ContextFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_word_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 paragraph_index=None,
                 segment_ids=None, # Deprecated due to context/question split
                 start_position=None,
                 end_position=None,
                 answer_mask=None,
                 doc_tokens=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_word_map = token_to_word_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.answer_mask = answer_mask
        self.paragraph_index = paragraph_index
        self.doc_tokens = doc_tokens


class QuestionFeatures(object):
    def __init__(self,
                 unique_id,
                 example_index,
                 tokens_,
                 input_ids,
                 input_mask,
                 segment_ids=None): # Deprecated due to context/question split
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens_ = tokens_
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class TfidfFeatures(object):
    def __init__(self,
                 pos_score,
                 neg_score):
        self.pos_score = pos_score
        self.neg_score = neg_score

    def __repr__(self):
        s = ""
        s += "scores %.2f/%.2f" % (self.pos_score, self.neg_score)
        return s


def read_squad_examples(input_file, return_answers, context_only=False, question_only=False,
                        draft=False, draft_num_examples=12):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    examples = []
    ans_cnt = 0
    no_ans_cnt = 0

    # Only word-based tokenization is peformed (whitespace based)
    for doc_idx, entry in enumerate(input_data):
        title = entry['title'][0] if type(entry['title']) == list else entry['title']
        assert type(title) == str

        for par_idx, paragraph in enumerate(entry["paragraphs"]):
            # Do not load context for question only
            if not question_only:
                paragraph_text = paragraph["context"]
                # Note that we use the term 'word' for whitespace based words, and 'token' for subtokens (for BERT input)
                doc_words, char_to_word_offset = context_to_words_and_offset(paragraph_text)

            # 1) Context only ends here
            if context_only:
                example = SquadExample(
                    doc_words=doc_words,
                    title=title,
                    doc_idx=doc_idx,
                    par_idx=par_idx
                )
                examples.append(example)

                if draft and len(examples) == draft_num_examples:
                    return examples
                continue

            # 2) Question only or 3) context/question pair
            else:
                for qa in paragraph["qas"]:
                    qas_id = str(qa["id"])
                    question_text = qa["question"]

                    # Noisy question skipping
                    if len(question_text.split(' ')) == 1:
                        logger.info('Skipping a single word question: {}'.format(question_text))
                        continue
                    if "I couldn't could up with another question." in question_text:
                        logger.info('Skipping a strange question: {}'.format(question_text))
                        continue

                    start_position = None
                    end_position = None
                    orig_answer_text = None

                    # For pre-processing that should return answers together
                    if return_answers:
                        assert type(qa["answers"]) == dict or type(qa["answers"]) == list, type(qa["answers"])
                        if type(qa["answers"]) == dict:
                            qa["answers"] = [qa["answers"]]

                        # No answers
                        if len(qa["answers"]) == 0:
                            orig_answer_text = ""
                            start_position = -1 # Word-level no-answer => -1
                            end_position = -1
                            no_ans_cnt += 1
                        # Answer exists
                        else:
                            answer = qa["answers"][0]
                            ans_cnt += 1

                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]

                            # Only add answers where the text can be exactly recovered from the context
                            actual_text = " ".join(doc_words[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                tokenization.whitespace_tokenize(orig_answer_text)) # word based tokenization
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                                continue

                    # Question only ends here
                    if question_only:
                        example = SquadExample(
                            qas_id=qas_id,
                            question_text=question_text)

                    # Context/question pair ends here
                    else:
                        example = SquadExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            paragraph_text=paragraph_text,
                            doc_words=doc_words,
                            orig_answer_text=orig_answer_text,
                            start_position=start_position,
                            end_position=end_position,
                            title=title,
                            doc_idx=doc_idx,
                            par_idx=par_idx)
                    examples.append(example)

                    if draft and len(examples) == draft_num_examples:
                        return examples

    # Testing for shuffled draft (should comment out above 'draft' if-else statements)
    if draft:
        random.shuffle(examples)
        logger.info(str(len(examples)) + ' were collected before draft for shuffling')
        return examples[:draft_num_examples]

    logger.info('Answer/no-answer stat: %d vs %d'%(ans_cnt, no_ans_cnt))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, return_answers, skip_no_answer,
                                 verbose=False, save_with_prob=False, msg="Converting examples"):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    features = []
    question_features = []

    for (example_index, example) in enumerate(tqdm(examples, desc=msg)):

        # Tokenize query into (sub)tokens
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # Creating a map between word <=> (sub)token
        tok_to_word_index = []
        word_to_tok_index = [] # word to (start of) subtokens
        all_doc_tokens = []
        for (i, word) in enumerate(example.doc_words):
            word_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(word)
            for sub_token in sub_tokens:
                tok_to_word_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -2 accounts for [CLS], [SEP]
        max_tokens_for_doc = max_seq_length - 2

        # Split sequence by max_seq_len with doc_stride, _DocSpan is based on tokens without [CLS], [SEP]
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_tok_offset = 0 # From all_doc_tokens

        # Get doc_spans with stride and offset
        while start_tok_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_tok_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_tok_offset, length=length))
            if start_tok_offset + length == len(all_doc_tokens):
                break
            start_tok_offset += min(length, doc_stride) # seems to prefer doc_stride always
            assert doc_stride < length, "length is no larger than doc_stride for {}".format(doc_spans)

        # Iterate each doc_span and make out_tokens
        for (doc_span_index, doc_span) in enumerate(doc_spans):

            # Find answer position based on new out_tokens
            start_position = None
            end_position = None

            # For no_answer, same (-1, -1) applies
            if example.start_position is not None and example.start_position < 0:
                assert example.start_position == -1 and example.end_position == -1
                start_position, end_position = NO_ANS, NO_ANS

            # For existing answers, find answers if exist
            elif return_answers:

                # Get token-level start/end position
                tok_start_position = word_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_words) - 1:
                    tok_end_position = word_to_tok_index[example.end_position + 1] - 1 # By backwarding from next word
                else:
                    assert example.end_position == len(example.doc_words) - 1
                    tok_end_position = len(all_doc_tokens) - 1

                # Improve answer span by subword-level
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

                # Throw away training samples without answers (due to doc_span split)
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if (tok_start_position < doc_start or tok_end_position < doc_start or
                    tok_start_position > doc_end or tok_end_position > doc_end):
                    if skip_no_answer:
                        continue
                    else:
                        # For NQ, only add this in 2% (50 times downsample)
                        if save_with_prob:
                            if np.random.randint(100) < 2:
                                start_position, end_position = NO_ANS, NO_ANS
                            else:
                                continue
                        else:
                            start_position, end_position = NO_ANS, NO_ANS

                # Training samples with answers
                else:
                    doc_offset = 1 # For [CLS]
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                    assert start_position >= 0 and end_position >= 0, (start_position, end_position)

            out_tokens = []  # doc
            out_tokens_ = [] # quesry
            out_tokens.append("[CLS]")
            out_tokens_.append("[CLS]")
            token_to_word_map = {} # The difference with tok_to_word_index is it includes special tokens
            token_is_max_context = {}

            # For query tokens, just copy and add [SEP]
            for token in query_tokens:
                out_tokens_.append(token)
            out_tokens_.append("[SEP]")

            # For each doc token, create token_to_word_map and is_max_context, and add to out_tokens
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_word_map[len(out_tokens)] = tok_to_word_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(out_tokens)] = is_max_context
                out_tokens.append(all_doc_tokens[split_token_index])
            out_tokens.append("[SEP]")

            # Convert to ids and masks
            input_ids = tokenizer.convert_tokens_to_ids(out_tokens)
            input_ids_ = tokenizer.convert_tokens_to_ids(out_tokens_)
            input_mask = [1] * len(input_ids)
            input_mask_ = [1] * len(input_ids_)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            while len(input_ids_) < max_query_length + 2: # +2 for [CLS], [SEP]
                input_ids_.append(0)
                input_mask_.append(0)
            assert len(input_ids_) == max_query_length + 2
            assert len(input_mask_) == max_query_length + 2

            # Printing for debug
            if example_index < 1 and verbose:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in out_tokens]))
                logger.info("q tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in out_tokens_]))
                logger.info("token_to_word_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_word_map)]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                if return_answers:
                    answer_text = " ".join(out_tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

            # Append feature
            features.append(
                ContextFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=out_tokens,
                    token_to_word_map=token_to_word_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    start_position=start_position,
                    end_position=end_position))
            question_features.append(
                QuestionFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    tokens_=out_tokens_,
                    input_ids=input_ids_,
                    input_mask=input_mask_))

            # Check validity of answer
            if return_answers:
                if start_position <= NO_ANS:
                    assert start_position == NO_ANS and end_position == NO_ANS, (start_position, end_position)
                else:
                    assert out_tokens[start_position:end_position+1] == \
                            all_doc_tokens[tok_start_position:tok_end_position+1]
                    orig_text, start_pos, end_pos = get_final_text_(
                        example, features[-1], start_position, end_position, True, False)
                    phrase = orig_text[start_pos:end_pos]
                    try:
                        assert phrase == example.orig_answer_text
                    except Exception as e:
                        # print('diff ans [%s]/[%s]'%(phrase, example.orig_answer_text))
                        pass
            unique_id += 1

    return features, question_features


def convert_questions_to_features(examples, tokenizer, max_query_length=None):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    question_features = []

    for (example_index, example) in enumerate(tqdm(examples, desc='Converting questions')):

        query_tokens = tokenizer.tokenize(example.question_text)
        if max_query_length is None:
            max_query_length = len(query_tokens)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        for _ in enumerate(range(1)):
            tokens_ = []
            tokens_.append("[CLS]")
            for token in query_tokens:
                tokens_.append(token)
            tokens_.append("[SEP]")

            input_ids_ = tokenizer.convert_tokens_to_ids(tokens_)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask_ = [1] * len(input_ids_)

            # Zero-pad up to the sequence length.
            while len(input_ids_) < max_query_length + 2:
                input_ids_.append(0)
                input_mask_.append(0)

            assert len(input_ids_) == max_query_length + 2
            assert len(input_mask_) == max_query_length + 2

            if example_index < 1:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in query_tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids_]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask_]))

            question_features.append(
                QuestionFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    tokens_=tokens_,
                    input_ids=input_ids_,
                    input_mask=input_mask_))
            unique_id += 1

    return question_features


def convert_documents_to_features(examples, tokenizer, max_seq_length, doc_stride):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    features = []

    for (example_index, example) in enumerate(tqdm(examples, desc='Converting documents')):

        # Creating a map between word <=> (sub)token
        tok_to_word_index = []
        word_to_tok_index = [] # word to (start of) subtokens
        all_doc_tokens = []
        for (i, word) in enumerate(example.doc_words):
            word_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(word)
            for sub_token in sub_tokens:
                tok_to_word_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -2 accounts for [CLS], [SEP]
        max_tokens_for_doc = max_seq_length - 2

        # Split sequence by max_seq_len with doc_stride, _DocSpan is based on tokens without [CLS], [SEP]
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_tok_offset = 0 # From all_doc_tokens

        # Get doc_spans with stride and offset
        while start_tok_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_tok_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_tok_offset, length=length))
            if start_tok_offset + length == len(all_doc_tokens):
                break
            start_tok_offset += min(length, doc_stride) # seems to prefer doc_stride always
            assert doc_stride < length, "length is no larger than doc_stride for {}".format(doc_spans)

        # Iterate each doc_span and make out_tokens
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            out_tokens = []  # doc
            out_tokens.append("[CLS]")
            token_to_word_map = {} # The difference with tok_to_word_index is it includes special tokens
            token_is_max_context = {}

            # For each doc token, create token_to_word_map and is_max_context, and add to out_tokens
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_word_map[len(out_tokens)] = tok_to_word_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(out_tokens)] = is_max_context
                out_tokens.append(all_doc_tokens[split_token_index])
            out_tokens.append("[SEP]")

            # Convert to ids and masks
            input_ids = tokenizer.convert_tokens_to_ids(out_tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            # Printing for debug
            if example_index < 1 and doc_span_index < 1:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in out_tokens]))
                logger.info("token_to_word_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_word_map)]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))

            # Append feature
            features.append(
                ContextFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=out_tokens,
                    token_to_word_map=token_to_word_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask))
            unique_id += 1

    return features


def context_to_words_and_offset(context):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_words = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in context:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_words.append(c)
            else:
                doc_words[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_words) - 1)

    return doc_words, char_to_word_offset


def inject_noise(input_ids, input_mask,
                 clamp=False, clamp_prob=0.5, min_len=0, max_len=300, pad=0,
                 replace=False, replace_prob=0.3, unk_prob=0.1, vocab_size=30522, unk=100, min_id=999,
                 shuffle=False, shuffle_prob=0.2):
    input_ids = input_ids[:]
    input_mask = input_mask[:]
    if clamp and random.random() < clamp_prob:
        len_ = sum(input_mask) - 2
        new_len = random.choice(range(min_len, max_len + 1))
        if new_len < len_:
            input_ids[new_len + 1] = input_ids[len_ + 1]
            for i in range(new_len + 2, len(input_ids)):
                input_ids[i] = pad
                input_mask[i] = 0

    len_ = sum(input_mask) - 2
    if replace:
        for i in range(1, len_ + 1):
            if random.random() < replace_prob:
                if random.random() < unk_prob:
                    new_id = unk
                else:
                    new_id = random.choice(range(min_id, vocab_size))
                input_ids[i] = new_id

    if shuffle:
        for i in range(1, len_ + 1):
            if random.random() < shuffle_prob:
                new_id = random.choice(input_ids[1:len_ + 1])
                input_ids[i] = new_id

    return input_ids, input_mask


def inject_noise_to_neg_features(features,
                                 clamp=False, clamp_prob=1.0, min_len=0, max_len=300, pad=0,
                                 replace=False, replace_prob=1.0, unk_prob=1.0, vocab_size=30522, unk=100, min_id=999,
                                 shuffle=False, shuffle_prob=1.0):
    features = copy.deepcopy(features)
    input_ids = features.input_ids
    input_mask = features.input_mask
    if clamp and random.random() < clamp_prob:
        len_ = sum(input_mask) - 2
        new_len = random.choice(range(min_len, min(len_, max_len) + 1))
        input_ids[new_len + 1] = input_ids[len_ + 1]
        for i in range(new_len + 2, len(input_ids)):
            input_ids[i] = pad
            input_mask[i] = 0

    len_ = sum(input_mask) - 2
    if replace:
        for i in range(1, len_ + 1):
            if random.random() < replace_prob:
                if random.random() < unk_prob:
                    new_id = unk
                else:
                    new_id = random.choice(range(min_id, vocab_size))
                input_ids[i] = new_id

    if shuffle:
        for i in range(1, len_ + 1):
            if random.random() < shuffle_prob:
                new_id = random.choice(input_ids[1:len_ + 1])
                input_ids[i] = new_id

    return features


def inject_noise_to_neg_features_list(features_list, noise_prob=1.0, **kwargs):
    out = [inject_noise_to_neg_features(features, **kwargs) if random.random() < noise_prob
           else features for features in features_list]
    return out


def sample_similar_questions(examples, features, question_emb_file, cuda=False):
    with h5py.File(question_emb_file, 'r') as fp:
        ids = []
        mats = []
        for id_, mat in fp.items():
            # Skip sparse
            if 'sparse' in id_ or '_input_ids' in id_:
                continue
            ids.append(id_)
            mats.append(mat[:])
        id2idx = {id_: idx for idx, id_ in enumerate(ids)}
        large_mat = np.concatenate(mats, axis=0)
        large_mat = torch.tensor(large_mat).float()
        if cuda:
            large_mat = large_mat.to(torch.device('cuda'))
        """
        sim = large_mat.matmul(large_mat.t())
        sim_argsort = (-sim).argsort(dim=1).cpu().numpy()
        """

        id2features = collections.defaultdict(list)
        for feature in features:
            id_ = examples[feature.example_index].qas_id
            id2features[id_].append(feature)

        sampled_features = []
        for feature in tqdm(features, desc='sampling'):
            example = examples[feature.example_index]
            example_tup = (example.title, example.doc_idx, example.par_idx)
            id_ = example.qas_id
            idx = id2idx[id_]
            similar_feature = None
            sim = (large_mat.matmul(large_mat[idx:idx+1, :].t()).squeeze(1))
            sim_argsort = (-sim).argsort(dim=0).cpu().numpy()
            for target_idx in sim_argsort:
                target_features = id2features[ids[target_idx]]
                for target_feature in target_features:
                    target_example = examples[target_feature.example_index]
                    target_tup = (target_example.title, target_example.doc_idx, target_example.par_idx)
                    if example_tup != target_tup:
                        similar_feature = target_feature
                        break
                if similar_feature is not None:
                    break

            assert similar_feature is not None
            sampled_features.append(similar_feature)
        return sampled_features


def compute_tfidf(train_examples, pos_features, neg_features, q_features, data_dir=None, tfidf_path=None):
    # Load doc ranker
    ranker = TfidfDocRanker(tfidf_path=tfidf_path, strict=False)
    qpar_cache = {} # used for tfidf score cache

    # Iterate each feature and compute pos/neg scores
    tfidf_features = []
    for pos_feature, neg_feature, q_feature in tqdm(zip(pos_features, neg_features, q_features), desc='tfidf compute'):
        q_text = ' '.join(q_feature.tokens_[1:-1])
        pos_text = ' '.join(pos_feature.tokens[1:-1])
        neg_text = ' '.join(neg_feature.tokens[1:-1])

        pos_score = None
        neg_score = None
        if q_text + '[SEP]' + pos_text in qpar_cache:
            pos_score = qpar_cache[q_text + '[SEP]' + pos_text]
        if q_text + '[SEP]' + neg_text in qpar_cache:
            neg_score = qpar_cache[q_text + '[SEP]' + neg_text]
        if pos_score is not None and neg_score is not None:
            tfidf_features.append(
                TfidfFeatures(
                    pos_score=pos_score,
                    neg_score=neg_score,
                )
            )
            continue

        # If any one of them is missing, compute them
        q_spvec = ranker.text2spvec(q_text)
        if pos_score is None:
            pos_spvec = ranker.text2spvec(pos_text)
            pos_score = (q_spvec*pos_spvec.T)[0,0],
            qpar_cache[q_text + '[SEP]' + pos_text] = pos_score
        if neg_score is None:
            neg_spvec = ranker.text2spvec(neg_text)
            neg_score = (q_spvec*neg_spvec.T)[0,0],
            qpar_cache[q_text + '[SEP]' + neg_text] = neg_score

        tfidf_features.append(
            TfidfFeatures(
                pos_score=pos_score,
                neg_score=neg_score,
            )
        )

    return tfidf_features
