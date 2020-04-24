import json
import argparse
import torch
import tokenization
import os
import random
import numpy as np
import requests
import logging
import math
import ssl
import best
import copy
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from time import time
from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from requests_futures.sessions import FuturesSession
from tqdm import tqdm
from collections import namedtuple

from serve_utils import load_caches, parse_example, get_cached, get_search
from train_query import train_query_encoder
from modeling import BertConfig
from query_encoder import QueryEncoder
from tfidf_doc_ranker import TfidfDocRanker
from run_natkb import check_diff
from pre import SquadExample, convert_questions_to_features
from post import convert_question_features_to_dataloader, get_question_results
from mips_phrase import MIPS
from mips_sent import MIPS_SENT
from eval_utils import normalize_answer, f1_score, exact_match_score, drqa_exact_match_score, drqa_regex_match_score,\
                       drqa_metric_max_over_ground_truths, drqa_normalize

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
nltk.download('punkt')
nltk.download('stopwords')


class NaturalKB(object):
    def __init__(self, args):
        self.args = args
        # IP and Ports
        self.base_ip = args.base_ip
        self.query_port = args.query_port
        self.doc_port = args.doc_port
        self.index_port = args.index_port

        # Saved objects
        self.mips = None

    def load_query_encoder(self, device, args):
        # Configure paths for query encoder serving
        vocab_path = os.path.join(args.metadata_dir, args.vocab_name)
        bert_config_path = os.path.join(
            args.metadata_dir, args.bert_config_name.replace(".json", "") + "_" + args.bert_model_option + ".json"
        )

        # Load pretrained QueryEncoder
        bert_config = BertConfig.from_json_file(bert_config_path)
        model = QueryEncoder(bert_config, use_biobert=args.use_biobert, hard_em=args.hard_em)
        if args.parallel:
            model = torch.nn.DataParallel(model)
        model.to(device)
        state = torch.load(args.query_encoder_path, map_location=device)
        model.load_state_dict(state['model'], strict=False)
        check_diff(model.state_dict(), state['model'])
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=not args.do_case)

        logger.info('Model loaded from %s' % args.query_encoder_path)
        logger.info('Number of model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
        return model, tokenizer

    def get_question_dataloader(self, questions, tokenizer, batch_size):
        question_examples = [SquadExample(qas_id='qs', question_text=q) for q in questions]
        query_features = convert_questions_to_features(
            examples=question_examples,
            tokenizer=tokenizer,
            max_query_length=64
        )
        question_dataloader = convert_question_features_to_dataloader(
            query_features,
            fp16=False, local_rank=-1,
            predict_batch_size=batch_size
        )
        return question_dataloader, question_examples, query_features

    def serve_query_encoder(self, query_port, args):
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer = self.load_query_encoder(device, args)

        # Define query to vector function
        def query2vec(queries):
            question_dataloader, question_examples, query_features = self.get_question_dataloader(
                queries, tokenizer, batch_size=24
            )
            query_encoder.eval()
            question_results = get_question_results(
                question_examples, query_features, question_dataloader, device, query_encoder
            )
            outs = []
            for qr_idx, question_result in enumerate(question_results):
                for ngram in question_result.sparse.keys():
                    question_result.sparse[ngram] = question_result.sparse[ngram].tolist()
                out = (
                    question_result.start.tolist(), question_result.end.tolist(),
                    question_result.sparse, question_result.input_ids
                )
                outs.append(out)
            return outs

        # Serve query encoder
        app = Flask(__name__)
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        @app.route('/batch_api', methods=['POST'])
        def batch_api():
            batch_query = json.loads(request.form['query'])
            outs = query2vec(batch_query)
            return jsonify(outs)

        logger.info(f'Starting QueryEncoder server at {self.get_address(query_port)}')
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(query_port)
        IOLoop.instance().start()

    def load_phrase_index(self, args, dump_only=False, sent_search=False):
        if self.mips is not None:
            return self.mips

        # Configure paths for index serving
        phrase_dump_dir = os.path.join(args.dump_dir, args.phrase_dir)
        tfidf_dump_dir = os.path.join(args.dump_dir, args.tfidf_dir)
        index_dir = os.path.join(args.dump_dir, args.index_dir)
        index_path = os.path.join(index_dir, args.index_name)
        idx2id_path = os.path.join(index_dir, args.idx2id_name)
        max_norm_path = os.path.join(index_dir, 'max_norm.json')

        # Load mips
        mips_init = MIPS if not sent_search else MIPS_SENT
        mips = mips_init(
            phrase_dump_dir=phrase_dump_dir,
            tfidf_dump_dir=tfidf_dump_dir,
            start_index_path=index_path,
            idx2id_path=idx2id_path,
            max_norm_path=max_norm_path,
            doc_rank_fn={
                'index': self.get_doc_scores, 'top_docs': self.get_top_docs, 'doc_meta': self.get_doc_meta,
                'spvec': self.get_q_spvecs
            },
            cuda=args.cuda, dump_only=dump_only
        )
        return mips

    def best_search(self, query, kcw_path=None):
        t0 = time()
        # Type filter
        ent_types = [
            "gene", "drug", "chemical compound", "target", "disease",
            "toxin", "transcription factor", "mirna", "pathway", "mutation"
        ]
        query_type = "All Entity Type"
        for ent_type in ent_types:
            if ent_type in query:
                query_type = ent_type
                break

        # Stopwords and kcw filter
        stop_words = set(stopwords.words('english') + ['?'] + ['Why', 'What', 'How', 'Where', 'When', 'Who'])
        entity_set = [
            'COVID-19', 'SARS-CoV-2', 'hypertension', 'diabetes', 'heart', 'disease', 'obese', 'death',
            'HCoV-19', 'HCoV', 'coronavirus', 'symptoms', 'incubation', 'periods', 'period', 'quarantine',
            'asymptomatic', 'transmissions', 'fecal', 'excretion', 'decline', 'Wuhan', 'mortality',
            'patients', 'stay', 'reproduction', 'risk', 'factor', 'factors', 'pregnancy', 'interval', 'absent',
            'reported', 'length', 'diagnosed', 'United', 'States', 'isolated', 'CDC', 'WHO', 'vaccine',
            'negative', 'animals', 'airbone', 'spread', 'blood', 'sanitizer', 'controlled', 'illness', 'friends',
        ]
        query_tokens = word_tokenize(query)
        new_query = ''
        for idx, query_token in enumerate(query_tokens):
            if query_token not in stop_words:
                if query_token in entity_set:
                    new_query += query_token + ' '

        # Get BEST result
        q = best.BESTQuery(new_query, noAbsTxt=False, filterObjectName=query_type)
        r = best.getRelevantBioEntities(q)

        # No result
        if len(r) == 1 and r[0]['rank'] == 0 and len(r[0].keys()) == 1:
            t1 = time()
            return {'ret': [], 'time': int(1000 * (t1 - t0))}

        parsed_result = {
            'context': '',
            'title': '',
            'doc_idx': None,
            'start_idx': 0,
            'end_idx': 0,
            'score': 0,
            'metadata': {
                'pubmed_id': ''
            },
            'answer': ''
        }
        outs = []
        for r_idx, r_ in enumerate(r):
            parsed_result['context'] = r_['abstracts'][0]
            parsed_result['score'] = r_['score']
            parsed_result['answer'] = r_['entityName']
            parsed_result['metadata'] = self.get_doc_meta(r_['PMIDs'][0])
            if len(parsed_result['metadata']) == 0:
                parsed_result['metadata']['pubmed_id'] = int(r_['PMIDs'][0])
            outs.append(copy.deepcopy(parsed_result))

        t1 = time()
        return {'ret': outs, 'time': int(1000 * (t1 - t0))}

    def serve_phrase_index(self, index_port, args):
        if index_port == '80':
            app = Flask(__name__, static_url_path='/static', static_folder="static",
                template_folder="templates")
            app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
            CORS(app)
            @app.before_request
            def before_request():
                if request.url.startswith('http://'):
                    url = request.url.replace('http://', 'https://', 1)
                    code = 301
                    return redirect(url, code=code)
            http_server = HTTPServer(WSGIContainer(app))
            http_server.listen(index_port)
            IOLoop.instance().start()
            return

        dev_str = '_dev' if args.develop else ''
        args.examples_path = os.path.join(f'static{dev_str}', args.examples_path)
        args.google_examples_path = os.path.join(f'static{dev_str}', args.google_examples_path)
        args.kcw_examples_path = os.path.join(f'static{dev_str}', args.kcw_examples_path)
        args.top100_covid_examples_path = os.path.join(f'static{dev_str}', args.top100_covid_examples_path)
        args.top100_google_examples_path = os.path.join(f'static{dev_str}', args.top100_google_examples_path)
        args.top10_kcw_examples_path = os.path.join(f'static{dev_str}', args.top10_kcw_examples_path)

        # Load mips
        self.mips = self.load_phrase_index(args, sent_search=args.sent_search)
        app = Flask(__name__, static_url_path='/static' + dev_str, static_folder="static" + dev_str,
            template_folder="templates" + dev_str)
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        # From serve_utils
        cached_set = load_caches(args)
        index_example_set, search_examples, inverted_examples, query_entity_ids = parse_example(args)

        def batch_search(batch_query, max_answer_length=20, start_top_k=1000, mid_top_k=100, top_k=10, doc_top_k=5,
                         nprobe=64, sparse_weight=0.05, search_strategy='hybrid', aggregate=False):
            t0 = time()
            outs, _ = self.embed_query(batch_query)()
            start = np.concatenate([out[0] for out in outs], 0)
            end = np.concatenate([out[1] for out in outs], 0)
            sparse_uni = [out[2]['1'][1:len(out[3])+1] for out in outs]
            sparse_bi = [out[2]['2'][1:len(out[3])+1] for out in outs]
            input_ids = [out[3] for out in outs]
            query_vec = np.concatenate([start, end, [[1]]*len(outs)], 1)

            rets = self.mips.search(
                query_vec, (input_ids, sparse_uni, sparse_bi), q_texts=batch_query, nprobe=nprobe,
                doc_top_k=doc_top_k, start_top_k=start_top_k, mid_top_k=mid_top_k, top_k=top_k,
                search_strategy=search_strategy, filter_=args.filter, max_answer_length=max_answer_length,
                sparse_weight=sparse_weight, aggregate=aggregate
            )
            t1 = time()
            out = {'ret': rets, 'time': int(1000 * (t1 - t0))}
            return out


        @app.route('/')
        def index():
            return render_template(f'index.html')

        @app.route('/.well-known/pki-validation/2E756828539AA07F898AE7E17BA4A104.txt')
        def pki_validation():
            return app.send_static_file('ssl/2E756828539AA07F898AE7E17BA4A104.txt')

        @app.route('/files/<path:path>')
        def static_files(path):
            return app.send_static_file('files/' + path)

        @app.route('/cached_example', methods=['GET'])
        def cached_example():
            start_time = time()
            q_id = request.args['q_id']
            res, query, query_info = get_cached(search_examples, q_id, query_entity_ids, cached_set)
            latency = time() - start_time
            latency = format(latency, ".3f")
            return render_template(f'cached.html', latency=latency, res=res, query=query, query_info=query_info)

        @app.route('/search', methods=['GET'])
        def search():
            query = request.args['query']
            params = {
                'strat': request.args['strat'] if 'strat' in request.args else 'dense_first',
                'm_a_l': (int(request.args['max_answer_length']) if 'max_answer_length' in request.args
                          else int(args.max_answer_length)),
                't_k': int(request.args['top_k']) if 'top_k' in request.args else int(args.top_k),
                'n_p': int(request.args['nprobe']) if 'nprobe' in request.args else int(args.nprobe),
                'd_t_k': int(request.args['doc_top_k']) if 'doc_top_k' in request.args else int(args.doc_top_k),
                's_w': (float(request.args['sparse_weight']) if 'sparse_weight' in request.args
                        else float(args.sparse_weight)),
                'a_g': (request.args['aggregate'] == 'True') if 'aggregate' in request.args else True
            }
            logger.info(f'{params["strat"]} search strategy is used.')

            out = batch_search(
                [query],
                max_answer_length = params['m_a_l'],
                top_k = params['t_k'],
                nprobe = params['n_p'],
                search_strategy=params['strat'], # [DFS, SFS, Hybrid]
                doc_top_k = params['d_t_k'],
                sparse_weight = params['s_w'],
                aggregate = params['a_g']
            )
            out['ret'] = out['ret'][0]
            out['ret'] = out['ret'][:3] # Get top 3 only
            b_out = self.best_search(query, kcw_path=args.kcw_examples_path)

            res, query, query_info = get_search(inverted_examples, search_examples, query_entity_ids, query, out, b_out)
            return render_template(f'search.html', latency=out['time'],
                    res=res, query=query, query_info=query_info, params=params)

        # This one uses a default hyperparameters
        @app.route('/api', methods=['GET'])
        def api():
            query = request.args['query']
            strat = request.args['strat']
            out = batch_search(
                [query],
                max_answer_length=args.max_answer_length,
                top_k=args.top_k,
                nprobe=args.nprobe,
                search_strategy=strat,
                doc_top_k=args.doc_top_k
            )
            out['ret'] = out['ret'][0]
            return jsonify(out)

        @app.route('/batch_api', methods=['POST'])
        def batch_api():
            batch_query = json.loads(request.form['query'])
            max_answer_length = int(request.form['max_answer_length'])
            start_top_k = int(request.form['start_top_k'])
            mid_top_k = int(request.form['mid_top_k'])
            top_k = int(request.form['top_k'])
            doc_top_k = int(request.form['doc_top_k'])
            nprobe = int(request.form['nprobe'])
            sparse_weight = float(request.form['sparse_weight'])
            strat = request.form['strat']
            out = batch_search(
                batch_query,
                max_answer_length=max_answer_length,
                start_top_k=start_top_k,
                mid_top_k=mid_top_k,
                top_k=top_k,
                doc_top_k=doc_top_k,
                nprobe=nprobe,
                sparse_weight=sparse_weight,
                search_strategy=strat,
                aggregate=args.aggregate
            )
            return jsonify(out)

        @app.route('/get_examples', methods=['GET'])
        def get_examples():
            return render_template(f'example.html', res = index_example_set)

        @app.route('/set_query_port', methods=['GET'])
        def set_query_port():
            self.query_port = request.args['query_port']
            return jsonify(f'Query port set to {self.query_port}')

        if self.query_port is None:
            logger.info('You must set self.query_port for querying. You can use self.update_query_port() later on.')
        logger.info(f'Starting Index server at {self.get_address(index_port)}')
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(
            os.path.join(os.path.expanduser('~'), "github/natural-kb/certs/covidsearch_korea_ac_kr.crt"),
            os.path.join(os.path.expanduser('~'), "github/natural-kb/certs/covidsearch_korea_ac_kr.pvk"),
        )
        # http_server = HTTPServer(WSGIContainer(app), ssl_options=ssl_ctx)
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(index_port)
        IOLoop.instance().start()

    def serve_doc_ranker(self, doc_port, args):
        doc_ranker_path = os.path.join(args.dump_dir, args.doc_ranker_name)
        doc_ranker = TfidfDocRanker(doc_ranker_path, strict=False)
        app = Flask(__name__)
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        @app.route('/doc_index', methods=['POST'])
        def doc_index():
            batch_query = json.loads(request.form['query'])
            doc_idxs = json.loads(request.form['doc_idxs'])
            outs = doc_ranker.batch_doc_scores(batch_query, doc_idxs)
            logger.info(f'Returning {len(outs)} from batch_doc_scores')
            return jsonify(outs)

        @app.route('/top_docs', methods=['POST'])
        def top_docs():
            batch_query = json.loads(request.form['query'])
            top_k = int(request.form['top_k'])
            batch_results = doc_ranker.batch_closest_docs(batch_query, k=top_k)
            top_idxs = [b[0] for b in batch_results]
            top_scores = [b[1].tolist() for b in batch_results]
            logger.info(f'Returning from batch_doc_scores')
            return jsonify([top_idxs, top_scores])

        @app.route('/doc_meta', methods=['POST'])
        def doc_meta():
            pmid = request.form['pmid']
            doc_meta = doc_ranker.get_doc_meta(pmid)
            # logger.info(f'Returning {len(doc_meta)} metadata from get_doc_meta')
            return jsonify(doc_meta)

        @app.route('/text2spvec', methods=['POST'])
        def text2spvec():
            batch_query = json.loads(request.form['query'])
            q_spvecs = [doc_ranker.text2spvec(q, val_idx=True) for q in batch_query]
            q_vals = [q_spvec[0].tolist() for q_spvec in q_spvecs]
            q_idxs = [q_spvec[1].tolist() for q_spvec in q_spvecs]
            logger.info(f'Returning {len(q_vals), len(q_idxs)} q_spvecs')
            return jsonify([q_vals, q_idxs])

        logger.info(f'Starting DocRanker server at {self.get_address(doc_port)}')
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(doc_port)
        IOLoop.instance().start()

    def get_address(self, port):
        assert self.base_ip is not None and len(port) > 0
        return self.base_ip + ':' + port

    def embed_query(self, batch_query):
        emb_session = FuturesSession()
        r = emb_session.post(self.get_address(self.query_port) + '/batch_api', data={'query': json.dumps(batch_query)})
        def map_():
            result = r.result()
            emb = result.json()
            return emb, result.elapsed.total_seconds() * 1000
        return map_

    def embed_all_query(self, questions, batch_size=16):
        all_outs = []
        for q_idx in tqdm(range(0, len(questions), batch_size)):
            outs, _ = self.embed_query(questions[q_idx:q_idx+batch_size])()
            all_outs += outs
        start = np.concatenate([out[0] for out in all_outs], 0)
        end = np.concatenate([out[1] for out in all_outs], 0)

        # input ids are truncated (no [CLS], [SEP]) but sparse vals are not ([CLS] max_len [SEP])
        sparse_uni = [out[2]['1'][1:len(out[3])+1] for out in all_outs]
        sparse_bi = [out[2]['2'][1:len(out[3])+1] for out in all_outs]
        input_ids = [out[3] for out in all_outs]
        query_vec = np.concatenate([start, end, [[1]]*len(all_outs)], 1)
        logger.info(f'Query reps: {query_vec.shape}, {len(input_ids)}, {len(sparse_uni)}, {len(sparse_bi)}')
        return query_vec, input_ids, sparse_uni, sparse_bi

    def query(self, query, search_strategy='hybrid'):
        params = {'query': query, 'strat': search_strategy}
        res = requests.get(self.get_address(self.index_port) + '/api', params=params)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            outs = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for q {query}')
            logger.info(res.text)
        return outs

    def batch_query(self, batch_query, max_answer_length=20, start_top_k=1000, mid_top_k=100, top_k=10, doc_top_k=5,
                    nprobe=64, sparse_weight=0.05, search_strategy='hybrid'):
        post_data = {
            'query': json.dumps(batch_query),
            'max_answer_length': max_answer_length,
            'start_top_k': start_top_k,
            'mid_top_k': mid_top_k,
            'top_k': top_k,
            'doc_top_k': doc_top_k,
            'nprobe': nprobe,
            'sparse_weight': sparse_weight,
            'strat': search_strategy,
        }
        res = requests.post(self.get_address(self.index_port) + '/batch_api', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            outs = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for q {batch_query}')
            logger.info(res.text)
        return outs

    def get_doc_scores(self, batch_query, doc_idxs):
        post_data = {
            'query': json.dumps(batch_query),
            'doc_idxs': json.dumps(doc_idxs)
        }
        res = requests.post(self.get_address(self.doc_port) + '/doc_index', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            result = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for {doc_idxs}')
            logger.info(res.text)
        return result

    def get_top_docs(self, batch_query, top_k):
        post_data = {
            'query': json.dumps(batch_query),
            'top_k': top_k
        }
        res = requests.post(self.get_address(self.doc_port) + '/top_docs', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            result = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for {top_k}')
            logger.info(res.text)
        return result

    def get_doc_meta(self, pmid):
        post_data = {
            'pmid': pmid
        }
        res = requests.post(self.get_address(self.doc_port) + '/doc_meta', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            result = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for {pmid}')
            logger.info(res.text)
        return result

    def get_q_spvecs(self, batch_query):
        post_data = {'query': json.dumps(batch_query)}
        res = requests.post(self.get_address(self.doc_port) + '/text2spvec', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            result = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for q {batch_query}')
            logger.info(res.text)
        return result

    def update_query_port(self, query_port):
        params = {'query_port': query_port}
        res = requests.get(self.get_address(self.index_port) + '/set_query_port', params=params)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            outs = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for port {query_port}')
            logger.info(res.text)
        logger.info(outs)

    def load_qa_pairs(self, data_path, args):
        q_ids = []
        questions = []
        answers = []
        data = json.load(open(data_path))['data']
        for item in data:
            q_id = item['id']
            question = item['question'] # + '?' # For NQ
            answer = item['answers']
            q_ids.append(q_id)
            questions.append(question)
            answers.append(answer)
        questions = [query.replace('[MASK] .', '_') for query in questions] # For RE datasets (no diff)

        if args.draft:
            rand_idxs = np.random.choice(len(questions), 20, replace=False)
            q_ids = np.array(q_ids)[rand_idxs].tolist()
            questions = np.array(questions)[rand_idxs].tolist()
            answers = np.array(answers)[rand_idxs].tolist()
        logger.info(f'Sample Q ({q_ids[0]}): {questions[0]}, A: {answers[0]}')
        logger.info(f'Evaluating {len(questions)} questions from {args.test_path}')
        return q_ids, questions, answers

    def eval_inmemory(self, args):
        # Load dataset and encode queries
        qids, questions, answers = self.load_qa_pairs(args.test_path, args)
        query_vec, input_ids, sparse_uni, sparse_bi = self.embed_all_query(questions)

        # Load MIPS
        self.mips = self.load_phrase_index(args)

        # Search
        step = args.eval_batch_size
        predictions = []
        for q_idx in tqdm(range(0, len(questions), step)):
            result = self.mips.search(
                query_vec[q_idx:q_idx+step],
                (input_ids[q_idx:q_idx+step], sparse_uni[q_idx:q_idx+step], sparse_bi[q_idx:q_idx+step]),
                q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
                doc_top_k=args.doc_top_k, start_top_k=args.start_top_k, mid_top_k=args.mid_top_k, top_k=args.top_k,
                search_strategy=args.search_strategy, filter_=args.filter, max_answer_length=args.max_answer_length,
                sparse_weight=args.sparse_weight, aggregate=args.aggregate
            )
            prediction = [[ret['answer'] for ret in out] for out in result]
            predictions += prediction

        self.evaluate_results(predictions, qids, questions, answers, args)

    def eval_request(self, args):
        # Load dataset
        qids, questions, answers = self.load_qa_pairs(args.test_path, args)

        # Run batch_query and evaluate
        step = args.eval_batch_size
        predictions = []
        evidences = []
        for q_idx in tqdm(range(0, len(questions), step)):
            result = self.batch_query(
                questions[q_idx:q_idx+step],
                max_answer_length=args.max_answer_length,
                start_top_k=args.start_top_k,
                mid_top_k=args.mid_top_k,
                top_k=args.top_k,
                doc_top_k=args.doc_top_k,
                nprobe=args.nprobe,
                sparse_weight=args.sparse_weight,
                search_strategy=args.search_strategy,
            )
            prediction = [[ret['answer'] for ret in out][:3] for out in result['ret']]
            evidence = [[ret['context'][ret['sent_start']:ret['sent_end']] for ret in out][:3] for out in result['ret']]
            predictions += prediction
            evidences += evidence
        self.evaluate_results(predictions, qids, questions, answers, args, evidences=evidences)

    def evaluate_results(self, predictions, qids, questions, answers, args, evidences=None):
        # Filter if there's candidate
        if args.candidate_path is not None:
            candidates = set()
            with open(args.candidate_path) as f:
                for line in f:
                    line = line.strip().lower()
                    candidates.add(line)
            logger.info(f'{len(candidates)} candidates are loaded from {args.candidate_path}')
            topk_preds = [list(filter(lambda x: (x in candidates) or (x.lower() in candidates), a)) for a in predictions]
            topk_preds = [a if len(a) > 0 else [''] for a in topk_preds]
            predictions = topk_preds[:]
            top1_preds = [a[0] for a in topk_preds]
        else:
            predictions = [a if len(a) > 0 else [''] for a in predictions]
            top1_preds = [a[0] for a in predictions]
        no_ans = sum([a == '' for a in top1_preds])
        logger.info(f'no_ans/all: {no_ans}, {len(top1_preds)}')
        logger.info(f'Evaluating {len(top1_preds)} answers.')

        # Get em/f1
        f1s, ems = [], []
        for prediction, groundtruth in zip(top1_preds, answers):
            if len(groundtruth)==0:
                f1s.append(0)
                ems.append(0)
                continue
            f1s.append(max([f1_score(prediction, gt)[0] for gt in groundtruth]))
            ems.append(max([exact_match_score(prediction, gt) for gt in groundtruth]))
        final_f1, final_em = np.mean(f1s), np.mean(ems)
        logger.info('EM: %.2f, F1: %.2f'%(final_em * 100, final_f1 * 100))

        # Top 1/k em (or regex em)
        exact_match_topk = 0
        exact_match_top1 = 0
        f1_score_topk = 0
        f1_score_top1 = 0
        pred_out = {}
        for i in range(len(predictions)):
            # For debugging
            if i < 3:
                logger.info(f'{i+1}) {questions[i]}')
                logger.info(f'=> groudtruths: {answers[i]}, prediction: {predictions[i][:5]}')

            match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score
            em_topk = max([drqa_metric_max_over_ground_truths(
                match_fn, prediction, answers[i]
            ) for prediction in predictions[i]])
            em_top1 = drqa_metric_max_over_ground_truths(
                match_fn, top1_preds[i], answers[i]
            )
            exact_match_topk += em_topk
            exact_match_top1 += em_top1

            f1_topk = 0
            f1_top1 = 0
            if not args.regex:
                match_fn = lambda x, y: f1_score(x, y)[0]
                f1_topk = max([drqa_metric_max_over_ground_truths(
                    match_fn, prediction, answers[i]
                ) for prediction in predictions[i]])
                f1_top1 = drqa_metric_max_over_ground_truths(
                    match_fn, top1_preds[i], answers[i]
                )
                f1_score_topk += f1_topk
                f1_score_top1 += f1_top1

            pred_out[qids[i]] = {
                    'question': questions[i],
                    'answer': answers[i], 'prediction': predictions[i],
                    'evidence': evidences[i] if evidences is not None else '',
                    'em_top1': bool(em_top1), f'em_top{args.top_k}': bool(em_topk),
                    'f1_top1': f1_top1, f'f1_top{args.top_k}': f1_topk
            }
        total = len(predictions)
        exact_match_top1 = 100.0 * exact_match_top1 / total
        f1_score_top1 = 100.0 * f1_score_top1 / total
        logger.info({'exact_match_top1': exact_match_top1, 'f1_score_top1': f1_score_top1})
        exact_match_topk = 100.0 * exact_match_topk / total
        f1_score_topk = 100.0 * f1_score_topk / total
        logger.info({f'exact_match_top{args.top_k}': exact_match_topk, f'f1_score_top{args.top_k}': f1_score_topk})

        # Dump predictions
        if not os.path.exists('pred'):
            os.makedirs('pred')
        pred_path = os.path.join('pred', os.path.splitext(os.path.basename(args.test_path))[0] + '.pred')
        logger.info(f'Saving prediction file to {pred_path}')
        with open(pred_path, 'w') as f:
            json.dump(pred_out, f)

    def save_top_k(self, args):
        # Load dataset and encode queries
        q_ids, questions, _ = self.load_qa_pairs(args.test_path, args)
        query_vec, input_ids, sparse_uni, sparse_bi = self.embed_all_query(questions)

        # Load MIPS
        self.mips = self.load_phrase_index(args)
        args.kcw_examples_path = os.path.join('static', args.kcw_examples_path)

        # Search
        step = args.eval_batch_size
        predictions = []
        b_out = []
        for q_idx in tqdm(range(0, len(questions), step)):
            prediction = self.mips.search(
                query_vec[q_idx:q_idx+step],
                (input_ids[q_idx:q_idx+step], sparse_uni[q_idx:q_idx+step], sparse_bi[q_idx:q_idx+step]),
                q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
                doc_top_k=args.doc_top_k, start_top_k=args.start_top_k, mid_top_k=args.mid_top_k, top_k=args.top_k,
                search_strategy=args.search_strategy, filter_=args.filter, max_answer_length=args.max_answer_length,
                sparse_weight=args.sparse_weight, aggregate=args.aggregate
            )
            predictions += prediction
            b_out += [self.best_search(query, args.kcw_examples_path) for query in questions[q_idx:q_idx+step]]

        # Dump predictions
        if not os.path.exists('pred'):
            os.makedirs('pred')
        with open(os.path.join('pred', f'top{args.top_k}_{os.path.basename(args.test_path)}'), 'w') as f:
            json.dump({'data': {q: {'denspi': p, 'best': b} for q, p, b in zip(q_ids, predictions, b_out)}}, f, indent=2)
        print()

    def get_top_phrases(self, path, args):
        if os.path.exists(path):
            with open(path, 'r') as f:
                in_file = json.load(f)
                questions = []
                answers = []
                phrase_idxs = []
                for line in in_file:
                    questions.append(line['question'])
                    answers.append(line['answer'])
                    phrase_idxs.append(line['phrase_idx'])
            if args.draft:
                questions = questions[:18]
                answers = answers[:18]
                phrase_idxs = phrase_idxs[:18]

            # Subsample for NQ train
            if 'nq-open/trainnono' in path:
                questions = questions[:10000]
                answers = answers[:10000]
                phrase_idxs = phrase_idxs[:10000]
            logger.info(f'Loading {len(phrase_idxs)} top phrase idxs from {path}')
            return questions, answers, phrase_idxs

        # Load dataset and encode queries
        _, questions, answers = self.load_qa_pairs(args.test_path, args)
        query_vec, input_ids, sparse_uni, sparse_bi = self.embed_all_query(questions)

        # Load MIPS
        self.mips = self.load_phrase_index(args)

        # Search
        step = args.eval_batch_size
        phrase_idxs = []
        for q_idx in tqdm(range(0, len(questions), step)):
            phrase_idx = self.mips.search(
                query_vec[q_idx:q_idx+step],
                (input_ids[q_idx:q_idx+step], sparse_uni[q_idx:q_idx+step], sparse_bi[q_idx:q_idx+step]),
                q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
                doc_top_k=args.doc_top_k, start_top_k=args.start_top_k, mid_top_k=args.mid_top_k, top_k=args.top_k,
                search_strategy=args.search_strategy, filter_=args.filter, return_idxs=True,
                max_answer_length=args.max_answer_length, sparse_weight=args.sparse_weight, no_para=args.no_para
            )
            phrase_idxs += phrase_idx

        # Save top phrases
        with open(path, 'w') as f:
            out_file = []
            for question, answer, phrase_idx in zip(questions, answers, phrase_idxs):
                out_file.append({'question': question, 'answer': answer, 'phrase_idx': phrase_idx})
            json.dump(out_file, f)
        logger.info(f'Saving top phrase idxs to {path}')

        return questions, answers, phrase_idxs

    def get_phrase_vecs(self, phrase_idxs, answers, batch_size, args, questions=None):
        # Load MIPS
        self.mips = self.load_phrase_index(args, dump_only=True)

        # Use tokenizer for vocab index
        vocab_path = os.path.join(args.metadata_dir, args.vocab_name)
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=not args.do_case)

        # Get phrase and vectors
        for item_idx in range(0, len(answers), batch_size):
            b_answers = answers[item_idx:item_idx+batch_size]
            b_phrase_idxs = phrase_idxs[item_idx:item_idx+batch_size]
            if questions is not None:
                b_questions = questions[item_idx:item_idx+batch_size]
            for b_idx, phrase_idx in enumerate(b_phrase_idxs):
                while len(b_phrase_idxs[b_idx]) < args.top_k:
                    b_phrase_idxs[b_idx].append((phrase_idx[0][0], 0, 0, ''))
                b_phrase_idxs[b_idx] = b_phrase_idxs[b_idx][:args.top_k]
            flat_phrase_idxs = [phrase for phrase_idx in b_phrase_idxs for phrase in phrase_idx]
            doc_groups = [self.mips.get_doc_group(phrase_idx_[0]) for phrase_idx_ in flat_phrase_idxs]
            start_idxs = [int(phrase_idx_[1]) for phrase_idx_ in flat_phrase_idxs]
            end_idxs = [int(phrase_idx_[2]) for phrase_idx_ in flat_phrase_idxs]
            phrases = [phrase_idx_[3] for phrase_idx_ in flat_phrase_idxs]
            phrase_lens = [group['f2o_end'][end_idx] - group['f2o_start'][start_idx] + 1
                for group, start_idx, end_idx in zip(doc_groups, start_idxs, end_idxs)]

            start_vecs = self.mips.dequant(
                doc_groups[0], np.stack([group['start'][start_idx] for group, start_idx in zip(doc_groups, start_idxs)])
            )
            end_vecs = self.mips.dequant(
                doc_groups[0], np.stack([group['end'][end_idx] for group, end_idx in zip(doc_groups, end_idxs)])
            )
            spans = [group['span_logits'][:] for group in doc_groups]
            spans = np.stack([span[start_idx, min(phrase_len, len(span[start_idx])-1)]
                if phrase_len >= 0 else span[start_idx, 0]
                for span, start_idx, end_idx, phrase_len in zip(spans, start_idxs, end_idxs, phrase_lens)])

            # Find targets based on exact string match
            match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score
            # match_fn = drqa_regex_match_score if args.regex else (lambda x,y: f1_score(x,y)[0] >= 0.5) # bad scores
            targets = [[drqa_metric_max_over_ground_truths(match_fn, phrase[3], answer) for phrase in phrase_idx]
                for phrase_idx, answer in zip(b_phrase_idxs, b_answers)]
            targets = [[ii for ii, val in enumerate(target) if val] for target in targets]

            # For debugging
            '''
            phrase_ids = [group['input_ids'][:][group['f2o_start'][start_idx]-20:group['f2o_end'][end_idx]+1+20]
                for group, start_idx, end_idx in zip(doc_groups, start_idxs, end_idxs)]
            keys = [kk for kk in tokenizer.vocab.keys()]
            text = [' '.join([keys[id_] for id_ in phrase]) for phrase in phrase_ids]
            target_texts = [np.array(b_phrase_idxs[k])[targets[k]] for k in range(len(targets))]
            def print_ds(kk):
                print(b_questions[kk], b_answers[kk])
                print(targets[kk])
                print(target_texts[kk])
                for kk_ in targets[kk]:
                    print(text[100*kk + kk_])
            import pdb; pdb.set_trace()
            '''

            # Calculate paragraph start end location in sparse vector
            para_lens = [group['len_per_para'][:] for group in doc_groups]
            f2o_start = [group['f2o_start'][:] for group in doc_groups]
            para_bounds = [[(sum(para_len[:para_idx]), sum(para_len[:para_idx+1])) for
                            para_idx in range(len(para_len))] for para_len in para_lens]
            para_idxs = []
            for para_bound, start_idx, f2o in zip(para_bounds, start_idxs, f2o_start):
                para_bound = np.array(para_bound)
                curr_idx = ((f2o[start_idx] >= para_bound[:,0]) & (f2o[start_idx] < para_bound[:,1])).nonzero()[0][0]
                para_idxs.append(curr_idx)
            para_startend = [para_bound[para_idx] for para_bound, para_idx in zip(para_bounds, para_idxs)]

            # Get sparse vectors
            p_ids_tmp = [group['input_ids'] for group in doc_groups]
            p_unis_tmp = [group['sparse'] for group in doc_groups]
            p_bis_tmp = [group['sparse_bi'] for group in doc_groups]
            p_ids = [sparse_id[p_se[0]:p_se[1]]
                     for sparse_id, p_se in zip(p_ids_tmp, para_startend)]
            p_unis = [self.mips.dequant(doc_groups[0], sparse_val[start_idx,:p_se[1]-p_se[0]], attr='sparse')
                    for sparse_val, p_se, start_idx in zip(p_unis_tmp, para_startend, start_idxs)]
            p_bis = [self.mips.dequant(doc_groups[0], sparse_bi_val[start_idx,:p_se[1]-p_se[0]-1], attr='sparse')
                    for sparse_bi_val, p_se, start_idx in zip(p_bis_tmp, para_startend, start_idxs)]

            # Reshape
            batch_size = len(b_answers) # for final one
            start_vecs = np.reshape(start_vecs, (batch_size, args.top_k, -1))
            end_vecs = np.reshape(end_vecs, (batch_size, args.top_k, -1))
            spans = np.reshape(spans, (batch_size, args.top_k))
            p_ids = [p_ids[ll:ll+args.top_k] for ll in range(0, len(p_ids), args.top_k)]
            p_unis = [p_unis[ll:ll+args.top_k] for ll in range(0, len(p_unis), args.top_k)]
            p_bis = [p_bis[ll:ll+args.top_k] for ll in range(0, len(p_bis), args.top_k)]
            yield start_vecs, end_vecs, spans, p_ids, p_unis, p_bis, targets

            '''
            yield 0, 0, 0, 0, 0, 0, [[]]
            '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # QueryEncoder
    parser.add_argument('--metadata_dir', default='/nvme/jinhyuk/denspi/bert', type=str)
    parser.add_argument("--vocab_name", default='vocab.txt', type=str)
    parser.add_argument("--bert_config_name", default='bert_config.json', type=str)
    parser.add_argument("--bert_model_option", default='large_uncased', type=str)
    parser.add_argument("--parallel", default=False, action='store_true')
    parser.add_argument("--do_case", default=False, action='store_true')
    parser.add_argument("--use_biobert", default=False, action='store_true')
    parser.add_argument("--query_encoder_path", default='/nvme/jinhyuk/denspi/KR94373_piqa-nfs_1173/1/model.pt', type=str)
    parser.add_argument("--query_port", default='-1', type=str)

    # DocRanker
    parser.add_argument('--doc_ranker_name', default='docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
    parser.add_argument('--doc_port', default='-1', type=str)

    # PhraseIndex
    parser.add_argument('--dump_dir', default='/nvme/jinhyuk/denspi/1173_wikipedia_filtered')
    parser.add_argument('--phrase_dir', default='phrase')
    parser.add_argument('--tfidf_dir', default='tfidf')
    parser.add_argument('--index_dir', default='1048576_hnsw_SQ8')
    parser.add_argument('--index_name', default='index.faiss')
    parser.add_argument('--idx2id_name', default='idx2id.hdf5')
    parser.add_argument('--index_port', default='-1', type=str)
    parser.add_argument('--sent_search', default=False, action='store_true')

    # These can be dynamically changed.
    parser.add_argument('--max_answer_length', default=20, type=int)
    parser.add_argument('--start_top_k', default=1000, type=int)
    parser.add_argument('--mid_top_k', default=100, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--doc_top_k', default=5, type=int)
    parser.add_argument('--nprobe', default=256, type=int)
    parser.add_argument('--sparse_weight', default=0.05, type=float)
    parser.add_argument('--search_strategy', default='hybrid')
    parser.add_argument('--filter', default=False, action='store_true')
    parser.add_argument('--aggregate', default=False, action='store_true')
    parser.add_argument('--no_para', default=False, action='store_true')

    # Serving options
    parser.add_argument('--examples_path', default='queries/examples_covid_paraphrased.json')
    parser.add_argument('--google_examples_path', default='queries/examples_google_query.json')
    parser.add_argument('--kcw_examples_path', default='queries/examples_covid_eval_kcw.json')
    parser.add_argument('--top100_covid_examples_path', default='queries/top100_covid_paraphrased.json')
    parser.add_argument('--top100_google_examples_path', default='queries/top100_google_query.json')
    parser.add_argument('--top10_kcw_examples_path', default='queries/top10_covid_eval_kcw.json')
    parser.add_argument('--develop', default=False, action='store_true')

    # Training (query_encoder)
    parser.add_argument('--train_path', default=None)
    parser.add_argument('--train_batch_size', default=10, type=int)
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--hard_em', default=False, action='store_true')
    parser.add_argument('--model_save_dir', default='models')

    # Evaluation
    parser.add_argument('--test_path', default='datasets/open-qa/squad/dev-v1.1_preprocessed_sampled.json')
    parser.add_argument('--candidate_path', default=None)
    parser.add_argument('--regex', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', default=10, type=int)
    parser.add_argument('--top_phrase_path', default='top_phrases.json')

    # Run mode
    parser.add_argument('--base_ip', default='http://163.152.163.248')
    parser.add_argument('--run_mode', default='batch_query')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--seed', default=1992, type=int)
    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    natural_kb = NaturalKB(args)

    # Set ports
    # natural_kb.query_port = '9010'
    # natural_kb.doc_port = '9020'
    # natural_kb.index_port = '10001'

    if args.run_mode == 'q_serve':
        logger.info(f'Query address: {natural_kb.get_address(natural_kb.query_port)}')
        natural_kb.serve_query_encoder(args.query_port, args)

    elif args.run_mode == 'd_serve':
        logger.info(f'Doc address: {natural_kb.get_address(natural_kb.doc_port)}')
        natural_kb.serve_doc_ranker(args.doc_port, args)

    elif args.run_mode == 'p_serve':
        logger.info(f'Query address: {natural_kb.get_address(natural_kb.query_port)}')
        logger.info(f'Doc address: {natural_kb.get_address(natural_kb.doc_port)}')
        logger.info(f'Index address: {natural_kb.get_address(natural_kb.index_port)}')
        natural_kb.serve_phrase_index(args.index_port, args)

    elif args.run_mode == 'query':
        logger.info(f'Index address: {natural_kb.get_address(natural_kb.index_port)}')
        query = 'Which Lisp framework has been developed for image processing?'
        # query = ' Several genetic factors have been related to HIV-1 resistance'
        result = natural_kb.query(query)
        logger.info(f'Answers to a question: {query}')
        logger.info(f'{[r["answer"] for r in result["ret"]]}')

        # Reshape metadata
        for result_ in result['ret']:
            meta = result_['metadata']
            if 'context_entities' not in meta:
                meta['context_entities'] = {}
                continue
            context_ents = [ # Initialize
                {k: [v_ for v_ in v.values()] for k, v in val.items()} for val in meta['context_entities'].values()
            ][0]
            assert len(meta['context_entities']) == 1, 'Currently single para is supported'
            for ii in range(1, len(meta['context_entities'])):
                for k, v in meta['context_entities'][ii]:
                    context_ents[k] += v
            meta['context_entities'] = context_ents

    elif args.run_mode == 'batch_query':
        logger.info(f'Index address: {natural_kb.get_address(natural_kb.index_port)}')
        queries = [
            'Which Lisp framework has been developed for image processing?',
            'What are the 3 main bacteria found in human milk?',
            'Where did COVID-19 happen?'
        ]
        result = natural_kb.batch_query(
            queries,
            max_answer_length=args.max_answer_length,
            start_top_k=args.start_top_k,
            mid_top_k=args.mid_top_k,
            top_k=args.top_k,
            doc_top_k=args.doc_top_k,
            nprobe=args.nprobe,
            sparse_weight=args.sparse_weight,
            search_strategy=args.search_strategy,
        )
        for query, result in zip(queries, result['ret']):
            logger.info(f'Answers to a question: {query}')
            logger.info(f'{[r["answer"] for r in result]}')

    elif args.run_mode == 'save_top_k':
        logger.info(f'Query address: {natural_kb.get_address(natural_kb.query_port)}')
        logger.info(f'Doc address: {natural_kb.get_address(natural_kb.doc_port)}')
        natural_kb.save_top_k(args)

    elif args.run_mode == 'eval_inmemory':
        logger.info(f'Query address: {natural_kb.get_address(natural_kb.query_port)}')
        logger.info(f'Doc address: {natural_kb.get_address(natural_kb.doc_port)}')
        natural_kb.eval_inmemory(args)

    elif args.run_mode == 'eval_request':
        logger.info(f'Index address: {natural_kb.get_address(natural_kb.index_port)}')
        natural_kb.eval_request(args)

    elif args.run_mode == 'get_doc_scores':
        logger.info(f'Doc address: {natural_kb.get_address(natural_kb.doc_port)}')
        queries = [
            'What was the Yuan\'s paper money called?',
            'What makes a successful startup??',
            'On which date was Genghis Khan\'s palace rediscovered by archeaologists?',
            'To-y is a _ .'
        ]
        result = natural_kb.get_doc_scores(queries, [[36], [2], [31], [22222]])
        logger.info(result)
        result = natural_kb.get_top_docs(queries, 5)
        logger.info(result)
        result = natural_kb.get_doc_meta('29970463') # Only used when there's doc_meta
        logger.info(result)
        result = natural_kb.get_doc_meta('k COVID-ABS_7f8715_viral entry properties required for fitness in humans are lost through rapid genomic change during viral isolation') # Only used when there's doc_meta
        logger.info(result)
        result = natural_kb.get_q_spvecs(queries)
        logger.info(result)

    elif args.run_mode == 'get_top_phrases':
        logger.info(f'Query address: {natural_kb.get_address(natural_kb.query_port)}')
        logger.info(f'Doc address: {natural_kb.get_address(natural_kb.doc_port)}')
        questions, answers, phrase_idxs = natural_kb.get_top_phrases(args.top_phrase_path, args)
        recall = []
        for _, _, _, _, _, _, targets in tqdm(natural_kb.get_phrase_vecs(phrase_idxs, answers, args.eval_batch_size, args,
            questions=questions),
                total=math.ceil(len(answers)/args.eval_batch_size)):
            for target in targets:
                if len(target) > 0:
                    recall.append(1)
                else:
                    recall.append(0)
        logger.info(f'Getting phrase vecs with recall {sum(recall)/len(recall)*100:.2f}% done.')

    elif args.run_mode == 'train_query':
        train_query_encoder(natural_kb, args)

    else:
        raise NotImplementedError
