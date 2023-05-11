import os
import argparse
import glob
from collections import defaultdict
from pprint import pprint

from elasticsearch import Elasticsearch
from tqdm import tqdm
from dotenv import load_dotenv


def ngrams(tokens, n=3, stride=1):
    return zip(*[tokens[i::stride] for i in range(0, n)])


def get_text(item):
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        return item['text']


def compute_overlap_es(client, ds_gen, ng=13):
    def get_query(msg):
        return {
            "query": {
                "match_phrase": {
                    "text": msg
                }
            }
        }
    # print(client.count(index='mc4_ngrams_13_1', body=query))
    ng2index = {
        13: ['mc4_ngrams_13_1', 'runeberg_ngrams_13_1'],
        7: ['mc4_ngrams_7_1', 'runeberg_ngrams_7_1']
    }
    index_names = ng2index.get(ng)
    total, found_cnt = 0, defaultdict(int)
    less_than_n, total_n_grams = 0, 0
    for dct in tqdm(list(ds_gen)):
        text = get_text(dct)
        tokens = text.split()
        less_than_n += len(tokens) < ng
        total += 1
        for ngram in ngrams(tokens, n=ng):
            final_count = 0
            for index_name in index_names:
                res = client.count(
                    index=index_name,
                    body=get_query(" ".join(ngram))
                )
                final_count += res['count']
            total_n_grams += 1
            for k in (1, 10, 100, 1000, 10000):
                found_cnt[k] += final_count >= k
    return {
        'ng': ng,
        'total_texts': total,
        'texts_length_less_than_{}'.format(ng): less_than_n,
        'found_cutoff_cnt': found_cnt,
        'total_{}_grams'.format(ng): total_n_grams,
        'found_cutoff_freq': {k: v / total_n_grams for k, v in found_cnt.items()}
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="absa_imm", help="One of c4_val, talbanken, lines, swequad_mc") 
    parser.add_argument('-s', '--split-name', type=str, default='dev', help="test, dev or train set")
    parser.add_argument('-ng', '--ngrams', type=int, default=7, help='which n-grams to use')
    args = parser.parse_args()
    load_dotenv()

    ELASTIC_PASSWORD = os.getenv('ES_PWD')

    # Create the client instance
    client = Elasticsearch(
        "https://localhost:9200",
        ca_certs=os.path.join("perplexity", "ca.crt"),
        basic_auth=("elastic", ELASTIC_PASSWORD)
    )

    if args.dataset == 'c4_val':
        from dataset import C4Dataset
        files = glob.glob(
            os.path.join("ctrl_data", "c4", "c4-sv-validation.tfrecord-*.json.gz")
        )
        print(files)
        N = len(files)
        ds = C4Dataset(files, validation_set=True)
        print("Loaded mC4 validation set with {} entries".format(len(ds)))
    else:
        from perplexity.dataset import *
        ds = globals()['load_{}'.format(args.dataset)]()

    res = compute_overlap_es(client, ds, ng=args.ngrams)
    pprint(res)
