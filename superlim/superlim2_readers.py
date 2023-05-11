import os
import glob
import csv

import jsonlines as jsl


#
# This file will have the readers for the SuperLim dataset
# Each reader is a generator that yields a dictionary,
# where each dictionary is guaranteed to contain a "text" field.
#

SUPERLIM_DIR="SuperLim-2-2.0.4"

SOURCES = {
    "absaimm": glob.glob(os.path.join(SUPERLIM_DIR, "absabank-imm", "*.jsonl")), 
    "dalajged": glob.glob(os.path.join(SUPERLIM_DIR, "dalaj-ged-superlim", "*.jsonl")),
    "swediagnostics": glob.glob(os.path.join(SUPERLIM_DIR, "swediagnostics", "*.jsonl")), 
    "swefaq": glob.glob(os.path.join(SUPERLIM_DIR, "swefaq", "*.jsonl")),
    "swefracas": [os.path.join(SUPERLIM_DIR, "swefracas", "swefracas.tsv")],
    "sweparaphrase": glob.glob(os.path.join(SUPERLIM_DIR, "sweparaphrase", "*.jsonl")),
    "swewinogender": glob.glob(os.path.join(SUPERLIM_DIR, "swewinogender", "*.jsonl")),
    "swewinograd": glob.glob(os.path.join(SUPERLIM_DIR, "swewinograd", "*.jsonl")),
    "swedn": glob.glob(os.path.join(SUPERLIM_DIR, "swedn", "*.jsonl")),
    "swenli": glob.glob(os.path.join(SUPERLIM_DIR, "swenli", "*.jsonl")),
    "swewic": glob.glob(os.path.join(SUPERLIM_DIR, "swewic", "*.jsonl"))
}

def get_split_files(files, split_name):
    return [x for x in files if os.path.splitext(os.path.basename(x))[0].endswith(split_name)]

def read_sv_files(files, callback, delimiter=','):
    for fname in files:
        with open(fname) as f:
            sv_file = csv.reader(f, delimiter=delimiter)
            head = next(sv_file)

            for line in sv_file:
                if not line: continue
                data = dict(zip(head, line))
                yield from callback(data)


def read_jsl_files(files, callback):
    for fname in files:
        with jsl.open(fname) as f:
            for obj in f:
                yield from callback(obj)


def read_absaimm(files=None, split_name='train', for_training=True):
    def process_file(data):
        if for_training:
            yield {
                'text': "{} Känsloläge:".format(data['text']),
                'label': data['label']
            }
        else:
            yield data
    files = get_split_files(
        files if files else SOURCES["absaimm"],
        split_name
    )
    yield from read_jsl_files(files, process_file)


def read_dalajged(files=None, split_name='train', for_training=True):
    # train, dev, test
    def process_file(dct):
        if for_training:
            yield {
                "text": "{} Fråga: Är meningen grammatiskt korrekt?".format(dct['sentence']),
                "label": "Ja" if dct["label"] == "correct" else "Nej"
            }
        else:
            yield dct
    files = get_split_files(
        files if files else SOURCES["dalajged"],
        split_name
    )
    yield from read_jsl_files(files, process_file)


def read_swefaq(files=None, split_name='train', for_training=True):
    def process_file(data):
        if for_training:
            for i, x in enumerate(data["candidate_answers"]):
                yield {
                    'text': "Fråga: {} Svar: {} Passar?".format(
                        data['question'], x
                    ),
                    'label': 'Ja' if i == data['label'] else 'Nej'
                }
        else:
            for i, x in enumerate(data["candidate_answers"]):
                yield {
                    'question': data['question'],
                    'answer': x,
                    'label': 'Ja' if i == data['label'] else 'Nej'
                }
    files = get_split_files(
        files if files else SOURCES["swefaq"],
        split_name
    )
    yield from read_jsl_files(files, process_file)


def read_sweparaphrase(files=None, split_name='train', for_training=True):
    def process_file(obj):
        if for_training:
            yield {
                "text": "Mening 1: {} Mening 2: {} Likhet mellan meningar:".format(
                    obj["sentence_1"], obj["sentence_2"]
                ),
                "label": obj["label"]
            }
        else:
            yield {
                "s1": obj["sentence_1"],
                "s2": obj["sentence_2"],
                "score": obj["label"]
            }
    files = get_split_files(
        files if files else SOURCES["sweparaphrase"],
        split_name
    )
    yield from read_jsl_files(files, process_file)


def read_swewinograd(files=None, split_name='train', for_training=True):
    def process_file(obj):
        if for_training:
            yield {
                "text": "{} Fråga: Syftar '{}' till '{}'? Svar:".format(
                    obj["text"], obj["pronoun"]["text"], obj['candidate_antecedent']['text']
                ),
                "label": "Ja" if obj["label"] == "coreferring" else "Nej"
            }
        else:
            yield obj
    files = get_split_files(
        files if files else SOURCES["swewinograd"],
        split_name
    )
    yield from read_jsl_files(files, process_file)


# Not the official SuperLim formulation, just out of curiosity
def read_swewinograd_res(files=None, split_name='train', for_training=True):
    def process_file(obj):
        if obj["label"] == "coreferring":
            if for_training:
                yield {
                    "text": "{} Fråga: Vem syftar ordet '{}' till? Svar:".format(
                        obj["text"], obj["pronoun"]["text"]
                    ),
                    "label": obj['candidate_antecedent']['text']
                }
            else:
                yield obj
    files = get_split_files(
        files if files else SOURCES["swewinograd"],
        split_name
    )
    yield from read_jsl_files(files, process_file)


def read_swedn(files=None, split_name="train", for_training=True):
    def process_file(obj):
        if for_training:
            yield {
                "text": "{} Sammanfattning:".format(obj["article"]),
                "label": obj["summary"]
            }
        else:
            yield obj
    files = get_split_files(
        files if files else SOURCES["swedn"],
        split_name
    )
    yield from read_jsl_files(files, process_file)


def read_swewic(files=None, split_name='train', for_training=True):
    def process_file(obj):
        if for_training:
            yield {
                "text": "Text 1: {} Text 2: {} Fråga: Betyder ordet '{}' samma sak i båda fall? Svar:".format(
                    obj["first"]["context"],
                    obj['second']['context'],
                    obj['first']['word']['text']
                ),
                "label": "Ja" if obj["label"] == "same_sense" else "Nej"
            }
        else:
            yield obj
    files = get_split_files(
        files if files else SOURCES["swewic"],
        split_name
    )
    yield from read_jsl_files(files, process_file)


def read_swenli(files=None, split_name="train", for_training=True):
    files = files if files else SOURCES["swenli"]
    yield from read_swediagnostics(files, split_name, for_training)


#
# All SuperLim datasets below do NOT have a training set,
# which is why the default split_name is 'test'
#
def read_swediagnostics(files=None, split_name='test', for_training=True):
    def process_file(data):
        LABEL_DICT = {
            "contradiction": "Nej",
            "neutral": "Kanske",
            "entailment": "Ja"
        }
        if for_training:
            yield {
                "text": "Situation: {} Påstående: {} Fråga: Stämmer? Svar:".format(
                    data["premise"], data["hypothesis"]
                ),
                "label": LABEL_DICT[data["label"]]
            }
        else:
            yield {
                'premise': data["premise"],
                "hypothesis": data["hypothesis"],
                "label": data["label"]
            }
    files = get_split_files(
        files if files else SOURCES["swediagnostics"],
        split_name
    )
    yield from read_jsl_files(files, process_file)


def read_swefracas(files=None, split_name='test', for_training=True):
    if split_name != "test": return 0
    cid, cdata = None, {}
    files = files if files else SOURCES["swefracas"]
    for fname in files:
        with open(fname) as f:
            sv_file = csv.reader(f, delimiter='\t')
            head = next(sv_file)

            for line in sv_file:
                if not line: continue
                data = dict(zip(head, line))
                if cid is None:
                    cid = data['id']
                elif cid != data['id']:
                    if for_training:
                        prem = cdata['premiss']
                        prem = " ".join(prem) if isinstance(prem, list) else prem
                        yield {
                            'text': "Premiss: {} Fråga: {} Svar:".format(
                                prem, cdata["fråga"]
                            ),
                            'label': cdata['svar']
                        }
                    else:
                        yield cdata
                    cdata = {}
                attr, val = data['attribute'], data['value']
                if val:
                    if attr in cdata:
                        if not isinstance(cdata[attr], list):
                            cdata[attr] = [cdata[attr]]
                        cdata[attr].append(val)
                    else:
                        cdata[attr] = val
                cid = data['id']
            if cdata:
                if for_training:
                    prem = cdata['premiss']
                    prem = " ".join(prem) if isinstance(prem, list) else prem
                    yield {
                        'text': "Premiss: {} Fråga: {} Svar:".format(
                            prem, cdata["fråga"]
                        ),
                        'label': cdata['svar']
                    }
                else:
                    yield cdata


def read_swewinogender(files=None, split_name='test', for_training=True):
    def process_file(obj):
        if for_training:
            yield {
                "text": "Situation: {} Påstående: {} Fråga: Stämmer? Svar:".format(
                    obj["premise"], obj["hypothesis"]
                ),
                "label": "Ja" if obj["label"] == "entailment" else "Kanske"
            }
        else:
            yield obj
    files = get_split_files(
        files if files else SOURCES["swewinogender"],
        split_name
    )
    yield from read_jsl_files(files, process_file)

