import os
import gzip
import json
import glob
import argparse

import jsonlines as jsl

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders

from dataset import gzip_iterator
from control_codes import START_C_CODES, END_C_CODES

UNKNOWN_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="A path to folder containing C4 data archived as GZIP archives")
    args = parser.parse_args()

    tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN))

    trainer = BpeTrainer(
        special_tokens=[UNKNOWN_TOKEN, PAD_TOKEN] + list(START_C_CODES.values()) + list(END_C_CODES.values()),
        vocab_size=256000,
        continuing_subword_prefix="##",
        # end_of_word_suffix="$$",
        show_progress=True
    )

    tokenizer.pre_tokenizer = Whitespace()

    files = glob.glob(os.path.join(args.data, "c4-sv.tfrecord-*.json.gz"))
    Nf = len(files)

    tokenizer.train_from_iterator(gzip_iterator(files[:Nf//3]), trainer)
    tokenizer.decoder = decoders.WordPiece() # to treat '##' properly

    tokenizer.save("tokenizer.json")

    tok = Tokenizer.from_file("tokenizer.json")
    out = tok.encode(":nyheter: Jag heter Lasse Supersvensson")
    print(out.tokens)
    print(tok.decode(out.ids))
