import os
import time
import argparse
import json
import sys
import math
from collections import defaultdict
from pprint import pprint

import torch
import numpy as np
from transformers import (
    PreTrainedTokenizerFast, CTRLConfig, CTRLTokenizer, CTRLModel, CTRLLMHeadModel
)
from tqdm import tqdm

from train_tokenizer import PAD_TOKEN, UNKNOWN_TOKEN
from control_codes import START_C_CODES, END_C_CODES


def get_text_from_parts(parts, tok):
    total_text = ""

    Np = len(parts)
    for i, p in enumerate(parts):
        txt_ids, s_id, s_len = p
        total_text += tok.batch_decode(txt_ids[:,s_id:s_id+s_len])[0]
        if i != Np - 1:
            total_text += " >|< "
    return total_text


# taken from https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore") 

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help="Checkpoint path")
    parser.add_argument('-c', '--control', type=str, help="Control code")
    parser.add_argument('-n', '--num_samples', type=int, help="Number of samples to generate")
    parser.add_argument('-caf', '--ctrl-args-file', type=str, required=True, help="CTRL settings file")
    parser.add_argument('-l', '--length', type=int, default=256, help="Text length to be generated")
    parser.add_argument('-ws', '--window-size', type=int, default=128, help="The size of the left sliding window")
    parser.add_argument('-gd', '--generate-deterministic', action='store_true', help="Whether to generate in a deterministic fashion")
    parser.add_argument('-o', '--output', type=str, help="Name of the output directory")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctrl_args = torch.load(args.ctrl_args_file)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=ctrl_args.tokenizer_file
    )
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.unk_token = UNKNOWN_TOKEN

    model = CTRLLMHeadModel.from_pretrained(args.file, local_files_only=True)
    model.to(device)
    model.eval()

    text = ""
    if args.control:
        text = START_C_CODES[args.control]
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)

    LIMITS = {
        'default': ctrl_args.sequence_length - 1,
    }

    total_length = args.length - 1
    max_len = min(total_length, LIMITS['default'])
    
    eos_token_id = None
    if args.control:
        eos_token_id = tokenizer.convert_tokens_to_ids(END_C_CODES[args.control])

    combo = {
        'repetition_penalty': 1.0,
        'top_p': 0.9,
    }

    generation_kwargs = {
        "do_sample": True,
        "pad_token_id": tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    }

    if eos_token_id:
        generation_kwargs['eos_token_id'] = eos_token_id

    folder_name = os.path.join(args.output, args.control)
    if os.path.exists(folder_name):
        # Check which ones were already generated and how much need to be added
        temp_data = defaultdict(lambda: defaultdict(int))
        top_p_data = defaultdict(lambda: defaultdict(int))
        for fname in os.listdir(folder_name):
            with open(os.path.join(folder_name, fname)) as f:
                res = json.load(f)
                rep_p = res['repetition_penalty']
                t = res.get('temperature')
                top_p = res.get('top_p')

                if t is not None:
                    temp_data[rep_p][t] += 1
                else:
                    top_p_data[rep_p][top_p] += 1
    else:
        temp_data, top_p_data = None, None
        os.makedirs(folder_name)

    print("Working on {}".format(args.control))
    
    print("Trying {}".format(combo.items()))

    if top_p_data is None and temp_data is None:
        num_samples = args.num_samples
    else:
        rep_p = combo['repetition_penalty']
        if "temperature" in combo:
            t = combo['temperature']
            num_samples = args.num_samples - temp_data[rep_p][t]
        elif "top_p" in combo:
            top_p = combo['top_p']
            num_samples = args.num_samples - top_p_data[rep_p][top_p]
        else:
            num_samples = args.num_samples

    generation_kwargs.update(combo)
    sample_idx = 0
    for nsample in tqdm(range(num_samples)):
        output = model.generate(
            input_ids, max_new_tokens=args.length,
            early_stopping=True, **generation_kwargs
        )
        decoded_text = tokenizer.batch_decode(output)[0]
        total_generated = output.shape[1]

        if total_length > max_len:
            # means we generate more than 255 chars, so apply sliding window approach
            ws = args.window_size
            parts = [(output, 0, max_len)]

            total_generated = max_len
            left_to_generate = total_length - total_generated
            steps = math.ceil(left_to_generate / ws)

            for _ in range(steps):
                if output[-1,-1] == eos_token_id:
                    break

                prompt_ids = output[:,-ws:]
                left_to_generate = total_length - total_generated
                max_len = min(left_to_generate, LIMITS['default'] - ws)
                output = model.generate(
                    prompt_ids, max_new_tokens=max_len,
                    early_stopping=True, eos_token_id=eos_token_id,
                    **generation_kwargs
                )
                parts.append((
                    output,
                    ws,
                    max_len
                ))
                total_generated += output.shape[1]
                
            res = {
                'total_generated': total_generated,
                'text': get_text_from_parts(parts)
            }
        else:
            res = {
                'total_generated': total_generated,
                'text': decoded_text
            }
        res.update(combo)
        res["control"] = args.control
        fname = 'gen_{}_{}.json'.format(int(time.time()), sample_idx)
        json.dump(res, open(os.path.join(folder_name, fname), 'w'))
        sample_idx += 1
