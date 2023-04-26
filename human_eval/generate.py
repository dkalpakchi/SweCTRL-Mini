import os
import argparse
import math
import time
import json
from pprint import pprint

import torch
import transformers
import yaml
from tqdm import tqdm

from common import init_tokenizer
from control_codes import START_C_CODES, END_C_CODES


def print_gen_step(idx):
    wlen = 20
    if idx >= 0:
        print("{0} GENERATION STEP {1} {0}".format("="*wlen, idx))
    else:
        print("{0} GENERATION FINISHED {0}".format("="*wlen))


def print_prompt(pr):
    print("\n\tPROMPT: {}\n".format(pr))


def get_text_from_parts(parts, tok):
    total_text = ""

    Np = len(parts)
    for i, p in enumerate(parts):
        txt_ids, s_id, s_len = p
        total_text += tok.batch_decode(txt_ids[:,s_id:s_id+s_len])[0]
        if i != Np - 1:
            total_text += " >|< "
    return total_text


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore") 

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help="Checkpoint path")
    parser.add_argument('-c', '--control', type=str, help="Control code")
    parser.add_argument('-p', '--prompt', type=str, help="Prompt (optional)")
    parser.add_argument('-pf', '--prompt_file', type=str, help="YAML prompt file (optional)")
    parser.add_argument('-caf', '--ctrl-args-file', type=str, required=True, help="CTRL settings file")
    parser.add_argument('-l', '--length', type=int, default=10, help="Number of new tokens to be generated")
    parser.add_argument('-ws', '--window-size', type=int, default=128, help="The size of the left sliding window")
    parser.add_argument('-gd', '--generate-deterministic', action='store_true', help="Whether to generate in a deterministic fashion")
    parser.add_argument('-o', '--output', type=str, help="Name of the output directory")
    parser.add_argument('-n', '--num_samples', type=int, help="Number of samples to generate")
    args = parser.parse_args()

    print("Torch version: {}".format(torch.__version__))
    print("HF Transformers version: {}".format(transformers.__version__))
    
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = yaml.load(f, Loader=yaml.FullLoader)
    else:
        prompts = [args.prompt]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctrl_args = torch.load(args.ctrl_args_file)

    tokenizer = init_tokenizer(ctrl_args.tokenizer_file)

    model = transformers.CTRLLMHeadModel.from_pretrained(args.file, local_files_only=True)
    # print(model)
    print("Loaded a model with {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))
    model.to(device)
    model.eval()

    control_und = args.control
    args.control = args.control.replace("_", "/")
    for prompt in [y for x in prompts[control_und].values() for y in x]:
        args.prompt = prompt
        text = ""
        if args.control:
            text = START_C_CODES[args.control]
        if args.prompt:
            if text:
                text = "{} {}".format(text, args.prompt)
            else:
                text = args.prompt
        input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
        
        LIMITS = {
            'prompt': input_ids.shape[1],
            'default': ctrl_args.sequence_length - 1,
        }
        print("Prompt length: {} tokens".format(LIMITS['prompt']))
        total_length = 255
        max_len = min(total_length, LIMITS['default'])
        max_new_tokens = max_len - LIMITS['prompt']
        
        print_gen_step(1)
        print_prompt(text)
        
        eos_token_id = None
        if args.control:
            eos_token_id = tokenizer.convert_tokens_to_ids(END_C_CODES[args.control])

        if args.generate_deterministic:
            generation_kwargs = {
                "do_sample": False
            }
        else:
            generation_kwargs = {
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.4
            }

        generation_kwargs['max_length'] = None
        generation_kwargs['pad_token_id'] = tokenizer.pad_token_id
        if eos_token_id:
            generation_kwargs['eos_token_id'] = eos_token_id

        folder_name = os.path.join(args.output, control_und)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for nsample in tqdm(range(args.num_samples)):
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
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

                for _ in tqdm(range(steps)):
                    if output[-1,-1] == eos_token_id:
                        break

                    prompt_ids = output[:,-ws:]
                    left_to_generate = total_length - total_generated
                    if left_to_generate < 0:
                        break
                    max_new_tokens = min(left_to_generate, LIMITS['default'] - ws)
                    output = model.generate(
                        prompt_ids, max_new_tokens=max_new_tokens,
                        early_stopping=True, **generation_kwargs
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
            res["gen_args"] = generation_kwargs
            res["control"] = args.control
            res["prompt"] = args.prompt
            fname = 'gen_{}_{}.json'.format(int(time.time()), nsample)
            json.dump(res, open(os.path.join(folder_name, fname), 'w'))

