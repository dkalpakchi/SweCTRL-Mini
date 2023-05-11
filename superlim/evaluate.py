import os
import sys
import glob
import json
import argparse
import math
from operator import itemgetter
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Optional, Dict, Any

from tqdm import tqdm

import numpy as np
import torch
import transformers

from transformers import DataCollatorForTokenClassification
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList

from common import (
    ADDITIONAL_TOKENS_FILE, CONFIG_FILE, MODEL_FILE, CTRL_ARGS_FILE,
    FT_ARGS_FILE, FOLD_FILE, FOLD_IDS_FILE, CONTROL_CODES_FILE
)
from dataset import (
    SuperLimKFoldDataset, TokenizeTransform, AregLeftToRightTransform
)
from util import load_tokenizer, reassign_embeddings



def add_control_code(text, text_ids, control, **kwargs):
    cc_id = tokenizer(control, return_tensors='pt').input_ids.to(device)
    new_prompt = torch.cat((text_ids, cc_id), dim=1)
    outputs = model.generate(new_prompt, **kwargs)
    return tokenizer.batch_decode(outputs)[0], outputs


def cut_after(lst, token_id):
    res = []
    for x in lst:
        if x == token_id:
            break
        res.append(x)
    return res


def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, use_cache=None, **kwargs):
    # only last token for inputs_ids if past is defined in kwargs
    if past_key_values:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": use_cache, "attention_mask": attention_mask}


def update_model_kwargs_for_generation(
    self,
    outputs: transformers.utils.ModelOutput,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
) -> Dict[str, Any]:
    """
    Adapted from the original function from v4.21.1:
    https://github.com/huggingface/transformers/blob/f0d496828d3da3bf1e3c8fbed394d7847e839fa6/src/transformers/generation_utils.py#L603

    Dmytro (2023-03-23): removed attention handling that didn't handle padding correctly (now in greedy_search)
    """ 

    if "past_key_values" in outputs:
        model_kwargs["past"] = outputs.past_key_values
    elif "mems" in outputs:
        model_kwargs["past"] = outputs.mems
    elif "past_buckets_states" in outputs:
        model_kwargs["past"] = outputs.past_buckets_states
    else:
        model_kwargs["past"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    return model_kwargs

def greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,       # kept for compat, but unused
    output_hidden_states: Optional[bool] = None,    # kept for compat, but unused
    output_scores: Optional[bool] = None,           # kept for compat, but unused
    return_dict_in_generate: Optional[bool] = None, # kept for compat, but unused
    synced_gpus: Optional[bool] = False,            # kept for compat, but unused
    **model_kwargs,
) -> torch.LongTensor:
    """
    Adapted from the original function from v4.21.1:
    https://github.com/huggingface/transformers/blob/f0d496828d3da3bf1e3c8fbed394d7847e839fa6/src/transformers/generation_utils.py#L1538
    """
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    unfinished_sequences = input_ids.new(batch_size).fill_(1)
    cur_len = input_ids.shape[-1]
    num_gen_tokens = 0

    this_peer_finished = False  # used by synced_gpus only
    gen_seqs = None
    while True:
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # Dmytro: Forbid generating EOS token first
        if num_gen_tokens == 0:
            # shape is (B, V)
            next_token_logits[:,eos_token_id] = -1000000

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        if gen_seqs is None:
            gen_seqs = next_tokens.reshape(-1, 1)
        else:
            gen_seqs = torch.cat([gen_seqs, next_tokens.reshape(-1, 1)], dim=-1)

        # Dmytro [2023-03-23]:
        # Added handling of padded batches, which is incorrect in the original
        # HF transformers library implementation
        
        extra_pads = (torch.ones(batch_size, 1).to(input_ids.device) * pad_token_id).int()
        extra_attn = torch.zeros(batch_size, 1).int().to(input_ids.device)
        input_ids = torch.cat([input_ids, extra_pads], dim=-1)
        first_pads = (input_ids == pad_token_id).cumsum(-1)
        input_ids[first_pads == 1] = next_tokens
        model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], extra_attn], dim=-1)
        model_kwargs["attention_mask"][first_pads == 1] = 1
        
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        cur_len = cur_len + 1
        num_gen_tokens += 1

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, ()):
            break
    
    return gen_seqs


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore") 

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folds_folder', type=str, required=True, help="Path to the folder with all the folds in")
    parser.add_argument('-b', '--batch_size', type=int, default=2, help="Batch size for inference")
    parser.add_argument('-d', '--detailed', action='store_true')
    args = parser.parse_args()

    # Note: currently inferencing with batch size > 1 doesn't produce stable results
    #       i.e., one gets different results for the same input if that input is alone or in a batch of X
    #       the solution of this problem requires re-implementing parts of CTRL model in transformers
    #       to properly assign positional embeddings for the padding tokens
    # TEMP FIX: just fix batch size to 1
    args.batch_size = 1

    # Hot fixes
    transformers.generation_utils.GenerationMixin.greedy_search = greedy_search
    transformers.generation_utils.GenerationMixin._update_model_kwargs_for_generation = update_model_kwargs_for_generation
    transformers.CTRLLMHeadModel.prepare_inputs_for_generation = prepare_inputs_for_generation

    if "*" in args.folds_folder:
        print("Attempting to resolve pattern: {}".format(args.folds_folder))
        cands = glob.glob(args.folds_folder)
        Nc = len(cands)
        if Nc == 1:
            args.folds_folder = cands[0]
            print("Evaluating folds in {}".format(args.folds_folder))
        else:
            if Nc == 0:
                print("Found no matching folders!")
            else:
                print("Ambiguous folds folder path. Found multiple matching:\n{}".format(str(cands)))
            sys.exit(1)

    print("Torch version: {}".format(torch.__version__))
    print("HF Transformers version: {}".format(transformers.__version__))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctrl_args = torch.load(os.path.join(args.folds_folder, CTRL_ARGS_FILE))
    ft_args = torch.load(os.path.join(args.folds_folder, FT_ARGS_FILE))
    
    with open(os.path.join(args.folds_folder, CONTROL_CODES_FILE)) as f:
        add_cc = json.load(f)

    tokenizer, START_C_CODES, END_C_CODES = load_tokenizer(
        ctrl_args.tokenizer_file, add_cc['start'], add_cc['end']
    )
    
    sl_ds_name = ft_args.superlim_dataset
    max_new_tokens = 30 if sl_ds_name == 'swedn' else 10
    
    ds = SuperLimKFoldDataset(sl_ds_name, split_name='test', n_folds=-1)
    tokenized_train = TokenizeTransform(
        ds, tokenizer, START_C_CODES, END_C_CODES, delta=max_new_tokens
    )

    train_ds = AregLeftToRightTransform(
        tokenized_train, max_sequence_length=ctrl_args.sequence_length
    )
    all_ids = set(range(ds.size))
    print("Loaded dataset with {} datapoints".format(ds.size))

    eos_token_id = tokenizer.convert_tokens_to_ids(END_C_CODES[sl_ds_name])
    generation_kwargs = {
        "do_sample": False,
        'eos_token_id': eos_token_id,
        'max_length': None,
        'pad_token_id': tokenizer.pad_token_id
    }
    
    EVAL_FNAME = "eval_detailed.json" if args.detailed else "eval.json"
    
    folds = glob.glob(os.path.join(args.folds_folder, "fold_*"))
    fold_perf = []
    collator = DataCollatorForTokenClassification(tokenizer)
    with torch.no_grad():
        for fold in sorted(folds):
            model = transformers.CTRLLMHeadModel.from_pretrained(
                fold, local_files_only=True
            )
            
            # print(model)
            print("Loaded a model with {} parameters".format(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            ))
            model.to(device)
            model.eval()

            if ft_args.n_folds > 0:
                fold_ids = set(torch.load(os.path.join(fold, FOLD_IDS_FILE)))
                eval_fold_ids = all_ids - fold_ids
            else:
                eval_fold_ids = all_ids
            
            if args.detailed:
                stats = {
                    'data': [],
                    'class_dist': defaultdict(int),
                    'eos_token_id': eos_token_id,
                    'eos_token': END_C_CODES[sl_ds_name]
                }
            else:
                stats = {
                    'correct': 0,
                    'class_dist': defaultdict(int),
                    'total': 0
                }

            # Code for testing batch size of 1 manually
            # for eval_id in list(eval_fold_ids):
            #     dp = train_ds[eval_id].copy()
            #     # labels are all -100 for the input and then whatever needs
            #     # to be predicted + end-of-category symbol
            #     labels = [x for x in dp.pop("labels") if x != -100][:-1]
            #     label_str = tokenizer.decode(labels)
            #     output = model.generate(
            #         input_ids=torch.tensor([dp['input_ids']]).to(device),
            #         max_new_tokens=10,
            #         early_stopping=True, **generation_kwargs
            #     )
            #     N_input = len(dp['input_ids'])
            #     out_tokens = output[0]
            #     pred_tokens = out_tokens[N_input:]

            #     print(label_str, "-----", tokenizer.decode(pred_tokens))
            #     break


            # NOTE: the Subset does NOT provide a copy of the dataset
            #       however, the collator will make a copy of data
            eval_set = torch.utils.data.Subset(train_ds, list(eval_fold_ids))
            loader = torch.utils.data.DataLoader(
                eval_set, batch_size=args.batch_size, collate_fn=collator
            )

            for batch in tqdm(loader):
                # labels are all -100 for the input and then whatever needs
                # to be predicted + end-of-category symbol
                labels = batch.pop("labels")

                pred = model.generate(
                    torch.tensor(batch['input_ids']).to(device),
                    attention_mask=torch.tensor(batch['attention_mask']).to(device),
                    max_new_tokens=max_new_tokens,
                    early_stopping=True, **generation_kwargs
                )
                B1, B2 = pred.shape[0], labels.shape[0]
                assert B1 == B2, "The batch shapes differ ({} and {})!".format(B1, B2)
                
                for l, p in zip(labels.cpu().tolist(), pred.cpu().tolist()):
                    l = [l_id for l_id in l if l_id != -100]
                    p = cut_after(p, tokenizer.pad_token_id)
                    label_str = tokenizer.decode(l)
                    if args.detailed:
                        stats['data'].append({
                            'label': l,
                            'label_str': label_str,
                            'pred': p,
                            'pred_str': tokenizer.decode(p)
                        })
                    else:
                        stats['correct'] += l == p
                        stats['total'] += B1
                    stats['class_dist'][label_str] += 1
            stats['class_dist'] = dict(stats['class_dist'])
            fold_perf.append(stats)

            with open(os.path.join(fold, EVAL_FNAME), 'w') as f:
                json.dump(fold_perf, f)

