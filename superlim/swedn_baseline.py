import udon2
import stanza
import glob
from tqdm import tqdm

import superlim2_readers as superlim


if __name__ == '__main__':
    sv = stanza.Pipeline(lang='sv', processors='tokenize,lemma,pos,depparse')

    ds = superlim.read_swedn(split_name='test', for_training=False)

    for x in tqdm(ds):
        trees = sv(x['article'])
        dct = trees.sentences[0].to_dict()
        nodes = udon2.Importer.from_stanza([dct])
        print(nodes[0].get_subtree_text())
