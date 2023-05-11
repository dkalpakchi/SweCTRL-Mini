import os
import re
import sys
import glob
import gzip
import json
import random
import logging
import multiprocessing as mp
import copy
from collections import defaultdict, Counter
from itertools import permutations
from urllib.parse import urlparse

import jsonlines as jsl
import udon2

from torch.utils.data import Dataset

from tqdm import tqdm

from control_codes import START_C_CODES, END_C_CODES
from ctrl_data.c4.cleaning import wiki_clean, news_clean, unicode_clean
from ctrl_data.c4.cats_by_urls import UNCATEGORIZED_WEBSITES

IGNORE_INDEX = -100


BASE_FOLDER = "ctrl_data"
FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.ERROR, format=FORMAT)
logger = logging.getLogger(__name__)


def get_text(c, field):
    return c["extra"][field].replace("”", '"').strip() if "extra" in c and c["extra"] else c["text"].replace("”", '"').strip()


def get_item(seq, i, default):
    try:
        return seq[i]
    except IndexError:
        return default


class GenericDataset(Dataset):
    def __init__(self):
        self._data = []

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class TextualDataset(GenericDataset):
    def __init__(self, folder, title_in_first_row=False, control_code=None):
        super().__init__()
        for fname in tqdm(glob.glob("{}/**/*.txt".format(folder), recursive=True)):
            if title_in_first_row:
                f = open(fname)
                title = f.readline()
                text = f.read()

                self._data.append({
                    "text": text,
                    "control": control_code,
                    "title": title
                })
            else:
                text = open(fname).read()

                self._data.append({
                    "text": text,
                    "control": control_code
                })


class TextualDatasetFromJSON(GenericDataset):
    def __init__(self, fname, title_key='title', text_key="text", control_code=None):
        super().__init__()
        datapoints = json.load(open(fname))

        for dp in tqdm(datapoints):
            self._data.append({
                "text": dp[text_key],
                "control": control_code
            })
            if title_key and title_key in dp:
                self._data[-1]["title"] = dp[title_key]


class TextualDatasetFromIterator(GenericDataset):
    def __init__(self, iterator, cleaners=None):
        super().__init__()

        for text, code in iterator:
            if cleaners is not None:
                for cleaner in cleaners:
                    text = cleaner(text)

            if text.strip():
                self._data.append({
                    "text": text.strip(),
                    "control": code
                })


class CombinedDataset(GenericDataset):
    def __init__(self, *ds):
        super().__init__()

        for d in ds:
            self._data.extend(d._data)


class TokenizeTransform(GenericDataset):
    def __init__(self, dataset, tokenizer, return_ids=True, is_ddl=False,
        world_size=-1, local_rank=0, start_control_tokens=True, end_control_tokens=True):
        super().__init__()
        self.__tok = tokenizer
        self.__return_ids = return_ids
        self.__conv_func = int if self.__return_ids else str
        self.__is_ddl = is_ddl
        self.__world_size = world_size
        self.__local_rank = local_rank
        self.__start_ct = start_control_tokens
        self.__end_ct = end_control_tokens
        self.__encode(dataset)

    @property
    def tok(self):
        return self.__tok

    def __encode_text(self, dp, with_title=False):
        text, control = dp["text"], dp["control"]
        s_control = START_C_CODES[control] if self.__start_ct else None
        e_control = END_C_CODES[control] if self.__end_ct else None

        func_name = 'encode' if self.__return_ids else 'tokenize'
        func = getattr(self.__tok, func_name)

        encoded_text = func(text, add_special_tokens=False)
        encoded_s_control = func(s_control, add_special_tokens=False)
        encoded_e_control = func(e_control, add_special_tokens=False)

        if with_title:
            title = dp.get('title')

            if not title: return

            return {
                "prepend": func("{} {}".format(START_C_CODES["title"], title), add_special_tokens=False),
                "text": encoded_text,
                "s_control": encoded_s_control,
                "e_control": encoded_e_control
            }
        else:
            return {
                "text": encoded_text,
                "s_control": encoded_s_control,
                "e_control": encoded_e_control
            }

    def _encode_chunk(self, chunk, position=0):
        enc_data = []
        for dp in tqdm(chunk, position=position):
            if "text" in dp and "title" in dp:
                enc_data.append(self.__encode_text(dp, with_title=True))
            elif "text" in dp:
                enc_data.append(self.__encode_text(dp))
        return enc_data

    def _encode_chunk_wrapper(self, args):
        return self._encode_chunk(*args)

    def __encode(self, dataset):
        if self.__is_ddl:
            num_workers = (mp.cpu_count() - self.__world_size) // self.__world_size
        else:
            num_workers = mp.cpu_count() - 2

        N = len(dataset)
        k, m = divmod(N, num_workers)

        with mp.Pool(num_workers) as p:
            data = p.map(
                self._encode_chunk_wrapper,
                [(dataset[i*k+min(i, m):(i+1)*k+min(i+1, m)], (self.__local_rank * num_workers) + (i+1)) for i in range(num_workers)]
            )
            for x in data:
                self._data.extend(x)


class AregLeftToRightTransform(GenericDataset):
    def __init__(self, dataset=None, max_sequence_length=256, is_ddl=False, local_rank=-1):
        super().__init__()

        self.__is_ddl = is_ddl
        self.__local_rank = local_rank

        self.__max_seq_len = max_sequence_length

        if dataset:
            self.__encode(dataset)
        
    def __encode(self, dataset):
        assert hasattr(dataset, "tok"), "Run TokenizeTransform first"
        tok = dataset.tok

        position = self.__local_rank if self.__is_ddl else 0
        for dp in tqdm(dataset, position=position):
            prepend = dp.get("prepend", [])
            s_control_code = dp["s_control"] if dp["s_control"] else []
            e_control_code = dp["e_control"] if dp["e_control"] else []

            final_text = prepend + s_control_code + dp["text"] + e_control_code
            text_len = len(final_text)

            for i in range(0, text_len, self.__max_seq_len):
                inp = tok.prepare_for_model(
                    final_text[i:i+self.__max_seq_len],
                    return_token_type_ids=False
                )
                # From https://huggingface.co/docs/transformers/model_doc/ctrl
                # Note that the labels are shifted inside the model, i.e. you can set labels = input_ids 
                # Indices are selected in [-100, 0, ..., config.vocab_size]
                # All labels set to -100 are ignored (masked), the loss is only computed for labels in [0, ..., config.vocab_size]
                # v This is why we don't shift the labels here!
                inp["labels"] = inp["input_ids"]
                self._data.append(inp)

    def _to_list(self):
        return self._data

    def _from_list(self, lst):
        self._data = lst


def gzip_iterator(files, yield_codes=False, validation=False):
    AUTO_CLASSES_FNAME = "auto_classes.json"
    FNAME = "w2cat.json"
    OTHER_CAT = "other/other"

    w2cat_f = open(FNAME)
    auto_f = open(AUTO_CLASSES_FNAME)

    w2cat = json.load(w2cat_f)
    wauto = None if validation else json.load(auto_f)

    cleaners = {
        'wiki': [wiki_clean],
        'news': [news_clean],
        'default': [unicode_clean]
    }

    filter_urls_fname = "filtered_urls_all.json" if os.path.exists("filtered_urls_all.json") else "filtered_urls.json"
    filter_f = open(filter_urls_fname)
    filtered_urls = set([y for x in json.load(filter_f).values() for y in x])

    for path in tqdm(files):
        with gzip.open(path, "rt") as f:
            data = f.readlines()
            with jsl.Reader(data) as reader:
                for obj in reader:
                    netloc = urlparse(obj["url"]).netloc
                    is_manually_classified = netloc not in UNCATEGORIZED_WEBSITES and obj["url"] in w2cat
                    is_auto_classified = wauto.get(obj["url"], OTHER_CAT) != OTHER_CAT
                    if obj["url"] not in filtered_urls and (is_manually_classified or is_auto_classified):
                        text = obj['text']
                        if wauto is not None and not validation:
                            cat = wauto.get(obj["url"], w2cat.get(obj["url"]))
                        else:
                            cat = w2cat.get(obj["url"])

                        if cleaners is not None:
                            for cleaner in cleaners['default']:
                                text = cleaner(text)
                            
                            if cat.startswith("news"):
                                for cleaner in cleaners["news"]:
                                    text = cleaner(text)
                            else:
                                for cleaner in cleaners.get(cat, []):
                                    text = cleaner(text)

                        if yield_codes:
                            yield text, cat
                        else:
                            yield text
    w2cat_f.close()
    auto_f.close()
    filter_f.close()


def gzip_loader(files, position=0, validation=False):
    AUTO_CLASSES_FNAME = "auto_classes.json"
    FNAME = "w2cat_val.json" if validation else "w2cat.json"
    OTHER_CAT = "other/other"

    w2cat_f = open(FNAME)
    w2cat = json.load(w2cat_f)

    if validation:
        wauto = None
    else:
        auto_f = open(AUTO_CLASSES_FNAME)
        wauto = json.load(auto_f)

    cleaners = {
        'wiki': [wiki_clean],
        'news': [news_clean],
        'default': [unicode_clean]
    }

    if validation:
        filter_urls_fname = "filtered_urls_val.json"
    else:
        filter_urls_fname = "filtered_urls_all.json" if os.path.exists("filtered_urls_all.json") else "filtered_urls.json"
    filter_f = open(filter_urls_fname)
    filtered_urls = set([y for x in json.load(filter_f).values() for y in x])

    data = []
    for path in tqdm(files, position=position):
        with gzip.open(path, "rt") as f:
            data_lines = f.readlines()
            with jsl.Reader(data_lines) as reader:
                for obj in reader:
                    netloc = urlparse(obj["url"]).netloc
                    is_manually_classified = netloc not in UNCATEGORIZED_WEBSITES and obj["url"] in w2cat
                    is_auto_classified = not validation and wauto.get(obj["url"], OTHER_CAT) != OTHER_CAT
                    if obj["url"] not in filtered_urls and (is_manually_classified or is_auto_classified):
                        text = obj["text"]
                        if wauto is not None and not validation:
                            cat = wauto.get(obj["url"], w2cat.get(obj["url"]))
                        else:
                            cat = w2cat.get(obj["url"])
                        if cat is None: continue

                        if cleaners is not None:
                            for cleaner in cleaners['default']:
                                text = cleaner(text)

                            if cat.startswith("news"):
                                for cleaner in cleaners["news"]:
                                    text = cleaner(text)
                            else:
                                for cleaner in cleaners.get(cat, []):
                                    text = cleaner(text)

                        if text.strip():
                            data.append({
                                "text": text.strip(),
                                "control": cat
                            })
    w2cat_f.close()
    filter_f.close()
    
    if not validation:
        auto_f.close()
    return data


def gzip_loader_wrapper(args):
    return gzip_loader(*args)


def gzip_validation_loader_wrapper(args):
    return gzip_loader(*args, validation=True)


class C4Dataset(GenericDataset):
    def __init__(self, files, is_ddl=False, world_size=-1, cleaners=None, local_rank=-1, validation_set=False):
        super().__init__()

        N = len(files)

        if is_ddl:
            num_workers = min(N, (mp.cpu_count() - world_size) // world_size)
        else:
            num_workers = min(N, mp.cpu_count() // 2)
            local_rank = 0

        k, m = divmod(N, num_workers)
        
        with mp.Pool(num_workers) as p:
            data = p.map(
                gzip_validation_loader_wrapper if validation_set else gzip_loader_wrapper,
                [(files[i*k+min(i, m):(i+1)*k+min(i+1, m)], (local_rank * num_workers) + (i+1)) for i in range(num_workers)]
            )
            for x in data:
                self._data.extend(x)


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


def load_swequad_mc():
    data = []
    with open(os.path.join(BASE_FOLDER, "swequad_mc_training.json")) as f:
        train = json.load(f)
        for obj in train['data']:
            data.append(obj["context"])
    return data


if __name__ == '__main__':
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json"
    )

    files = glob.glob(os.path.join("data", "c4", "c4-sv.tfrecord-*.json.gz"))[:2]
    ds = C4Dataset(files)

    ds = TokenizeTransform(ds, tok, return_ids=True)

    ds = AregLeftToRightTransform(ds)

    for i, x in enumerate(ds):
        print(x)

        if i == 10:
            break
