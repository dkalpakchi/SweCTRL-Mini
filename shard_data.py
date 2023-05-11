import os
import math
import glob
import logging
import dataclasses as dc
import multiprocessing as mp

import torch
from transformers import (
    HfArgumentParser, PreTrainedTokenizerFast
)

from util import CtrlArguments
from dataset import (
    TextualDataset, CombinedDataset, TokenizeTransform,
    AregLeftToRightTransform, AregArbitraryOrderTransform,
    TextualDatasetFromIterator, gzip_iterator, QDataset, C4Dataset
)
from train_tokenizer import PAD_TOKEN, UNKNOWN_TOKEN

FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


@dc.dataclass
class ShardingArguments:
    nshards: int = dc.field(
        default=8,
        metadata={"help": "A number of data shards, should correspond to the number of GPUs"}
    )
    output_dir: str = dc.field(
        default="",
        metadata={"help": "The directory to output the shard files to"}
    )


if __name__ == '__main__':
    parser = HfArgumentParser((ShardingArguments, CtrlArguments))
    shard_args, ctrl_args = parser.parse_args_into_dataclasses()

    logger.info(shard_args)
    logger.info(ctrl_args)

    tok = PreTrainedTokenizerFast(
        tokenizer_file=ctrl_args.tokenizer_file
    )
    tok.pad_token = PAD_TOKEN
    tok.unk_token = UNKNOWN_TOKEN

    CHUNK_SIZE = 100

    all_files = glob.glob(os.path.join(ctrl_args.train_data, "c4-sv.tfrecord-*.json.gz"))
    num_files = len(all_files)
    total_chunks = math.ceil(num_files / CHUNK_SIZE)

    for chunk_id in range(0, total_chunks):
        logger.info("==> Processing chunk {} of {}".format(chunk_id + 1, total_chunks))
        files = all_files[chunk_id*CHUNK_SIZE:(chunk_id+1)*CHUNK_SIZE]

        train_d1 = C4Dataset(files)

        if chunk_id == 0:
            train_d2 = TextualDataset(
                "data/runeberg/processed", title_in_first_row=True, control_code="lit"
            )
            train_d = CombinedDataset(train_d2, train_d1)
        else:
            train_d = train_d1

        tokenized_train = TokenizeTransform(train_d, tok)

        train_ds = AregLeftToRightTransform(
            tokenized_train, max_sequence_length=ctrl_args.sequence_length
        )

        Nds = len(train_ds)
        shard_size = Nds // shard_args.nshards

        train_ds = train_ds._to_list()

        if not os.path.exists(shard_args.output_dir):
            os.makedirs(shard_args.output_dir)

        for i in range(shard_args.nshards):
            shard_i = train_ds[i * shard_size:(i+1) * shard_size]
            shard_path = os.path.join(shard_args.output_dir, "shard_{}_{}.pt".format(i, chunk_id))
            torch.save(shard_i, shard_path)
            logger.info("({}) Saved shard {} to {}".format(chunk_id, i, shard_path))

    num_workers = min(total_chunks, mp.cpu_count())

    logger.info("Starting merging...")
    for i in range(shard_args.nshards):
        files2merge = glob.glob(os.path.join(shard_args.output_dir, "shard_{}*.pt".format(i)))
        with mp.Pool(num_workers) as p:
            chunks = p.map(torch.load, files2merge)
        total_chunk = chunks[0]
        for c in chunks[1:]:
            total_chunk.extend(c)
        logger.info("Chunk {} is merged successfully!".format(i))
        torch.save(total_chunk, os.path.join(shard_args.output_dir, "shard_{}.pt".format(i)))
        logger.info("Chunk {} is saved successfully!".format(i))

