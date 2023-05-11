import os
import glob
import json

import udon2


BASE_FOLDER = "ctrl_data"


def load_conll(fname):
    data = []
    trees = udon2.Importer.from_conll_file(fname)
    for t in trees:
        data.append(t.get_subtree_text())
    return data


def load_talbanken():
    return load_conll(os.path.join(BASE_FOLDER, "sv_talbanken-ud-train.conllu"))


def load_lines():
    return load_conll(os.path.join(BASE_FOLDER, "sv_lines-ud-train.conllu"))


def load_suc3():
    base_folder = os.path.join(BASE_FOLDER, "SUC3.0", "corpus", "conll")
    files = glob.glob(os.path.join(base_folder, "*.conll"))

    data = []
    for fn in files:
        data.extend(load_conll(fn))
    return data


def load_swequad_mc():
    data = []
    with open(os.path.join(BASE_FOLDER, "swequad_mc_training.json")) as f:
        train = json.load(f)
        for obj in train['data']:
            data.append(obj["context"])
    return data
