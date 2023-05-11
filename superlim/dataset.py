import sys
import os
import glob
import multiprocessing as mp
from collections import defaultdict

import numpy as np

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import superlim2_readers as superlim


class GenericDataset(Dataset):
    def __init__(self):
        self._data = []

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def validate_folds(f):
    def wrapper(*args):
        self, k = args[0], args[1]
        if k < self.n_folds:
            return f(*args)
        else:
            raise ValueError("The requested fold is non-existent")
    return wrapper


class SuperLimKFoldDataset(GenericDataset):
    def __init__(self, ds_name, split_name=None, n_folds=20, fold_size=100):
        super().__init__()
        try:
            loader = getattr(superlim, "read_{}".format(ds_name))
        except AttributeError:
            sys.exit(1)

        self.ds_name = ds_name
        # each loader is a generator function producing dicts
        # each dict is guaranteed to have a "text" field
        labels = defaultdict(list)
        loader_kwargs = {
            'for_training': True
        }
        if split_name is not None:
            loader_kwargs['split_name'] = split_name
        for i, x in enumerate(loader(**loader_kwargs)):
            x["control"] = ds_name.lower()
            self._data.append(x)
            labels[x['label']].append(i)

        N_labels = len(labels.keys())        
        self.size = len(self._data)
        self.n_folds = n_folds
        self.fold_size = fold_size

        if n_folds > 0:
            np.random.seed(19867)
            if N_labels < 5:
                # If the number of labels is more than this, chances are
                # it's impossible to make the dataset balanced
                min_size = min([len(x) for x in labels.values()]) // 2
                apriori_bucket_size = fold_size // N_labels
                self.bucket_sizes = {
                    k: min(len(v) // 2, apriori_bucket_size)
                    for k, v in labels.items()
                }

                total_folds = apriori_bucket_size * N_labels
                max_k = None
                if self.fold_size != total_folds:
                    max_k, _ = max(self.bucket_sizes.items(), key=lambda x: x[1])
                    self.bucket_sizes[max_k] += self.fold_size - total_folds
                
                # Sampling with replacement, but it shouldn't matter much
                self.folds = [
                    np.tile(
                        np.random.choice(label_ids, size=(n_folds, self.bucket_sizes[k])),
                        int(np.ceil(apriori_bucket_size / self.bucket_sizes[k]))
                    )[:,:(self.bucket_sizes[k] if k == max_k else apriori_bucket_size)]
                    for k, label_ids in labels.items()
                ]
                self.folds = np.hstack(self.folds)
            else:
                self.bucket_sizes = None
                idx = np.arange(self.size).astype(int)
                # Sampling with replacement, but it shouldn't matter much
                self.folds = np.random.choice(idx, size=(n_folds, fold_size))

            print("Sampled folds shape:", self.folds.shape)

    @validate_folds
    def get_fold_ids(self, k):
        return self.folds[k]

    @validate_folds
    def get_fold(self, k):
        ds = GenericDataset()
        for i in self.folds[k]:
            ds._data.append(self._data[i])
        return ds
    
    def get_control_code(self, start=True):
        lc_name = self.ds_name.lower()
        suffix = "" if start else "$"
        return {
            lc_name: ":{}:{}".format(lc_name, suffix)
        }


class CombinedDataset(GenericDataset):
    def __init__(self, *ds):
        super().__init__()

        for d in ds:
            self._data.extend(d._data)


class TokenizeTransform(GenericDataset):
    def __init__(self, dataset, tokenizer, start_c_codes, end_c_codes,
                 return_ids=True, max_context_length=245, max_sequence_length=256, delta=5):
        super().__init__()
        self.__tok = tokenizer
        self.__return_ids = return_ids
        self.__conv_func = int if self.__return_ids else str
        self.__max_len = max_context_length
        self.__max_seq_len = max_sequence_length - delta
        self.__start_c_codes = start_c_codes
        self.__end_c_codes = end_c_codes
        self.__encode(dataset)

    @property
    def tok(self):
        return self.__tok

    def __encode_text(self, dp):
        text, control, label = dp["text"], dp["control"], str(dp["label"])
        s_control = self.__start_c_codes[control]
        e_control = self.__end_c_codes[control]

        func_name = 'encode' if self.__return_ids else 'tokenize'
        func = getattr(self.__tok, func_name)

        encoded_label = func(label, add_special_tokens=False)
        encoded_s_control = func(s_control, add_special_tokens=False)
        encoded_e_control = func(e_control, add_special_tokens=False)
        Nl, Ns, Ne = len(encoded_label), len(encoded_s_control), len(encoded_e_control)
        
        max_len = self.__max_len
        
        if max_len + Nl + Ns + Ne > self.__max_seq_len:
            max_len = self.__max_seq_len - Nl - Ns - Ne

        # if the text eceeeds the given context length, insert [...] in the middle
        # because at the end we typically have questions
        encoded_text = func(text, add_special_tokens=False)
        N_text = len(encoded_text)
        if N_text > max_len:
            skip_token = func("[...]", add_special_tokens=False)
            tail = int(max_len // 2)
            start_cut, end_cut = tail, N_text - tail
            encoded_text = encoded_text[:start_cut] + skip_token + encoded_text[end_cut:]

        return {
            "prepend": encoded_text,
            "text": encoded_label,
            "s_control": encoded_s_control,
            "e_control": encoded_e_control
        }

    def _encode_chunk(self, chunk, position=0):
        enc_data = []
        for dp in tqdm(chunk, position=position):
            enc_data.append(self.__encode_text(dp))
        return enc_data

    def _encode_chunk_wrapper(self, args):
        return self._encode_chunk(*args)

    def __encode(self, dataset):
        num_workers = mp.cpu_count() - 2

        N = len(dataset)
        k, m = divmod(N, num_workers)

        with mp.Pool(num_workers) as p:
            data = p.map(
                self._encode_chunk_wrapper,
                [(dataset[i*k+min(i, m):(i+1)*k+min(i+1, m)],  num_workers + (i+1)) for i in range(num_workers)]
            )
            for x in data:
                self._data.extend(x)


class AregLeftToRightTransform(GenericDataset):
    def __init__(self, dataset=None, max_sequence_length=256):
        super().__init__()

        self.__max_seq_len = max_sequence_length

        if dataset:
            self.__encode(dataset)
        
    def __encode(self, dataset):
        assert hasattr(dataset, "tok"), "Run TokenizeTransform first"
        tok = dataset.tok

        for dp in tqdm(dataset):
            prepend = dp.get("prepend", [])
            s_control_code = dp["s_control"] if dp.get("s_control", False) else []
            e_control_code = dp["e_control"] if dp.get("e_control", False) else []

            # s_control_code = text control code
            # prepend = text + 
            final_text = s_control_code + prepend + dp["text"] + e_control_code
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
                inp["labels"] = list(inp["input_ids"])
                # Ignore the things related to text, since we want to predict only the things we need!
                for j in range(len(prepend) + 1):
                    inp["labels"][j] = -100
                self._data.append(inp)

    def _to_list(self):
        return self._data

    def _from_list(self, lst):
        self._data = lst


if __name__ == "__main__":
    from util import load_tokenizer
    ds = SuperLimKFoldDataset("swefaq")
    fold = ds.get_fold(5)
     
    tok, START_C_CODES, END_C_CODES = load_tokenizer(
        "tokenizer.json",
        ds.get_control_code(),
        ds.get_control_code(start=False)
    )

    #sys.exit(1)
    ds = TokenizeTransform(fold, tok, START_C_CODES, END_C_CODES, return_ids=False)

    #ds = AregLeftToRightTransform(ds)

    for x in ds:
        print(x)
