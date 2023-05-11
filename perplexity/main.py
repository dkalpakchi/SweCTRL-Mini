# Based on the code from https://huggingface.co/docs/transformers/perplexity
import argparse
import glob
import os
import logging

import torch
import transformers
from transformers import (
    CTRLConfig, CTRLLMHeadModel,
    AutoTokenizer, AutoModel, AutoModelForCausalLM
)
from tqdm import tqdm

from common import init_tokenizer
from dataset import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help="Dataset name (for which there is a loader in data.py)")
    parser.add_argument('-m', '--model', type=str, required=True, help="Checkpoint path or HF model ID")
    parser.add_argument('-caf', '--ctrl-args-file', type=str, default="", help="CTRL settings file")
    parser.add_argument('-l', '--local', action='store_true', help="Whether to search for a model only among local files")
    parser.add_argument('-ov', '--output-value-only', action='store_true', help="whether to output just perplexity value")
    parser.add_argument('-s', '--stride', type=int, default=8, help='Stride')
    args = parser.parse_args()

    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Transformers version: {}".format(transformers.__version__))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctrl_args = None
    DATA_DIR = "ctrl_data"

    if args.data == 'swectrl_train':
        files = glob.glob(
            os.path.join(DATA_DIR, "c4", "swedish", "c4-sv.tfrecord-*.json.gz")
        )
        N = len(files)
        random.shuffle(files)
        train_d1 = C4Dataset(files[:N//10])
        logger.info("Loaded a subset of mC4 training set with {} entries".format(len(train_d1)))

        train_d2 = TextualDataset(  
            os.path.join(DATA_DIR, "runeberg", "processed"),
            title_in_first_row=True, control_code="lit"
        )
        ds = CombinedDataset(train_d1, train_d2)
        random.shuffle(ds._data)
    elif args.data == 'c4_val':
        files = glob.glob(
            os.path.join(DATA_DIR, "c4", "c4-sv-validation.tfrecord-*.json.gz")
        )
        print(files)
        N = len(files)
        ds = C4Dataset(files, validation_set=True)
        logger.info("Loaded mC4 validation set with {} entries".format(len(ds)))
    else:
        ds = globals()['load_{}'.format(args.data)]()
    logger.info("Loaded dataset with {} entries".format(len(ds)))

    if args.ctrl_args_file:
        ctrl_args = torch.load(args.ctrl_args_file)
        logger.info(ctrl_args)

        tokenizer = init_tokenizer(ctrl_args.tokenizer_file)

        model = CTRLLMHeadModel.from_pretrained(args.model, local_files_only=args.local)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local)

    logger.debug(tokenizer)
    logger.debug(model)
    model.to(device)
    model.eval()

    max_length = model.config.n_positions
    stride = args.stride

    nlls = []
    num_tokens = 0
    skipped = 0

    for item in tqdm(ds):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item['text']
        encodings = tokenizer(text, return_tensors="pt")
        text_len = encodings.input_ids.size(1)

        if text_len < 2:
            skipped += 1 
            continue

        prev_end_loc = 0
        for begin_loc in range(0, text_len, stride):
            end_loc = min(begin_loc + max_length, text_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == text_len:
                break
        num_tokens += end_loc
    
    ppl = torch.exp(torch.stack(nlls).sum() / num_tokens)
    
    if args.output_value_only:
        print(ppl.item())
    else:
        logger.info("Perplexity: {}".format(ppl.item()))
        logger.info("Skipped: {}".format(skipped))
