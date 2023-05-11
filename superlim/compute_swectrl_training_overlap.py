import os
import argparse
import glob
import json
from collections import defaultdict
from pprint import pprint

from elasticsearch import Elasticsearch
from tqdm import tqdm
from dotenv import load_dotenv

import superlim2_readers as superlim


def ngrams(tokens, n=3, stride=1):
    return zip(*[tokens[i::stride] for i in range(0, n)])


# Adapted from:
# https://stackoverflow.com/questions/28546253/how-to-create-request-body-for-python-elasticsearch-msearch
def msearch(client, search_arr):
    request = ''
    for each in search_arr:
        request += '{} \n'.format(json.dumps(each))
    return client.msearch(body = request)


def flatten(lst):
    res = []
    for x in lst:
        if isinstance(x, list):
            res.extend(flatten(x))
        else:
            res.append(x)
    return res

def compute_overlap_es_text(client, tokens, ng=13):
    def get_query(msg, for_msearch_api=False):
        q =  {
            "query": {
                "match_phrase": {
                    "text": msg
                }
            }
        }
        if for_msearch_api:
            q['size'] = 0
            q['track_total_hits'] = True
        return q

    # print(client.count(index='mc4_ngrams_13_1', body=query))
    ng2index = {
        13: ['mc4_ngrams_13_1', 'runeberg_ngrams_13_1'],
        7: ['mc4_ngrams_7_1', 'runeberg_ngrams_7_1']
    }
    index_names = ng2index.get(ng)
    total_ng, found_ng = 0, 0
    search_arr, res = {}, {}
    for index_name in index_names:
        search_arr[index_name] = []
        res[index_name] = []
    for ngram in ngrams(tokens, n=ng):
        total_ng += 1
        for index_name in index_names:
            str_ngram = " ".join(ngram)
            # req_head
            search_arr[index_name].append({'index': index_name})
            # req_body
            search_arr[index_name].append(get_query(str_ngram, for_msearch_api=True))
   
    for index_name in index_names:
        if search_arr[index_name]:
            res[index_name] = msearch(client, search_arr[index_name])
  
    if sum([len(x) for x in res.values()]) == 0:
        return {
            'cnt': {},
            'ng': total_ng
        }

    found_cnt = defaultdict(int)
    for resp in zip(*[x['responses'] for x in res.values() if x]):
        final_count = sum([r['hits']['total']['value'] for r in resp if r])
        for k in (1, 10, 100):
            found_cnt[k] += final_count >= k
    return {
        'cnt': found_cnt,
        'ng': total_ng
    }


def compute_overlap_es(client, ds_gen, ng=13):
    def get_query(msg):
        return {
            "query": {
                "match_phrase": {
                    "text": msg
                }
            }
        }
    N = 4511237  # number of documents in training data
    total, found_cnt = 0, defaultdict(int)
    less_than_n, total_n_grams = 0, 0
    for dct in tqdm(list(ds_gen)):
        if 'text' in dct:
            texts = [dct['text']]
        elif 'sentence' in dct:
            texts = [dct['sentence']]
        elif 'article' in dct:
            texts = [dct['article']]
        elif 'premiss' in dct and 'fråga' in dct:
            texts = [dct['premiss'], dct['fråga']]
        elif 'premise' in dct and 'hypothesis' in dct:
            texts = [dct['premise'], dct['hypothesis']]
        elif 'question' in dct and 'answer' in dct:
            texts = [dct['question'], dct['answer']]
        elif 's1' in dct and 's2' in dct:
            texts = [dct['s1'], dct['s2']]
        elif 'first' in dct and 'second' in dct:
            texts = [dct['first']['context'], dct['second']['context']]
        else:
            print(dct)
            break
       
        texts = flatten(texts)
        for t in texts:
            tokens = t.split()
            less_than_n += len(tokens) < ng
            total += 1
            res = compute_overlap_es_text(client, tokens, ng)
            total_n_grams += res['ng']

        for k, v in res['cnt'].items():
            found_cnt[k] += v

    return {
        'ng': ng,
        'total_texts': total,
        'texts_length_less_than_{}'.format(ng): less_than_n,
        'freq_texts_length_less_than_{}'.format(ng): less_than_n / total,
        'total_{}_grams'.format(ng): total_n_grams,
        'found_cutoff_cnt': found_cnt,
        'found_cutoff_freq': {k: v / total_n_grams for k, v in found_cnt.items()}
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="absa_imm", help="A SuperLim dataset") 
    parser.add_argument('-s', '--split-name', type=str, default='dev', help="test, dev or train set")
    parser.add_argument('-ng', '--ngrams', type=int, default=7, help='which n-grams to use')
    args = parser.parse_args()
    load_dotenv()

    ELASTIC_PASSWORD = os.getenv('ES_PWD')

    # Create the client instance
    client = Elasticsearch(
        "https://localhost:9200",
        ca_certs="ca.crt",
        basic_auth=("elastic", ELASTIC_PASSWORD),
        request_timeout=5000
    )

    data_loader = getattr(superlim, "read_{}".format(args.dataset))

    if data_loader:
        gen = data_loader(split_name=args.split_name, for_training=False)
        res = compute_overlap_es(client, gen, ng=args.ngrams)
        pprint(res)

    #print(client.collections['mc4_ngrams_8'].documents.search(search_parameters))
