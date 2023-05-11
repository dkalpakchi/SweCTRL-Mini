import os
import argparse
import glob
from collections import defaultdict
from pprint import pprint

from tqdm import tqdm

import superlim2_readers as superlim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="absa_imm", help="A SuperLim dataset") 
    parser.add_argument('-s', '--split-name', type=str, default='dev', help="test, dev or train set")
    args = parser.parse_args()

    data_loader = getattr(superlim, "read_{}".format(args.dataset))

    if data_loader:
        gen = data_loader(split_name=args.split_name)

        stats = {
            'total': 0
        }
        for x in gen:
            stats['total'] += 1
        pprint(stats)

    #print(client.collections['mc4_ngrams_8'].documents.search(search_parameters))
