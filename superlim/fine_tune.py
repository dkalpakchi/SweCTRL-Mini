import os
import sys
import json
import copy
import logging

from tqdm import tqdm

import torch

import transformers
from transformers import (
    CTRLConfig, CTRLModel, CTRLLMHeadModel,
    HfArgumentParser, TrainingArguments, Trainer,
    DataCollatorForTokenClassification, PreTrainedTokenizerFast
)

from transformers.trainer_utils import get_last_checkpoint

from dataset import (
    SuperLimKFoldDataset, CombinedDataset, TokenizeTransform,
    AregLeftToRightTransform
)
from util import CtrlArguments, FineTuningArguments, load_tokenizer


CONTROL_CODES_INIT_FNAME = "control_init.bin"


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    parser = HfArgumentParser((TrainingArguments, FineTuningArguments))
    train_args, ft_args = parser.parse_args_into_dataclasses()
    print(train_args)
    print(ft_args)
    print("Torch version: {}".format(torch.__version__))
    print("Transformers version: {}".format(transformers.__version__))
   
    torch.manual_seed(433494437)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    transformers.logging.set_verbosity_info()

    ctrl_args = torch.load(ft_args.ctrl_args_file)
 
    ds = SuperLimKFoldDataset(
        ft_args.superlim_dataset,
        n_folds=ft_args.n_folds,
        fold_size=ft_args.fold_size
    )

    add_start_codes = ds.get_control_code()
    add_end_codes = ds.get_control_code(start=False)
    tokenizer, START_C_CODES, END_C_CODES = load_tokenizer(
        ctrl_args.tokenizer_file, add_start_codes, add_end_codes
    )
    
    # We know for sure that here we will have singular tokens,
    # so we avoid using tokenizer directly in order to avoid the following warning:
    #    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    c_codes_init = {}
    for cc_dct in (add_start_codes, add_end_codes):
        for cc_v in cc_dct.values():
            c_codes_init[cc_v] = tokenizer.backend_tokenizer.token_to_id(cc_v)

    collator = DataCollatorForTokenClassification(tokenizer)
   
    is_kfold = ft_args.n_folds > 0
    out_dir = train_args.output_dir
    for i_fold in range(ft_args.n_folds if is_kfold else 1):
        train_args.output_dir = os.path.join(out_dir, "fold_{}".format(i_fold))
        
        torch.manual_seed(87178291199)
        model = CTRLLMHeadModel.from_pretrained(
            ft_args.ctrl_checkpoint, local_files_only=True
        )
        model.resize_token_embeddings(len(tokenizer)) 

        init_emb = {}
        for k, v in c_codes_init.items():
            init_emb[k] = copy.deepcopy(model.transformer.w(torch.LongTensor([v])).cpu().tolist())
        
        if is_kfold:
            train_d = ds.get_fold(i_fold)
        else:
            train_d = ds

        tokenized_train = TokenizeTransform(
            train_d, tokenizer, START_C_CODES, END_C_CODES
        )

        train_ds = AregLeftToRightTransform(
            tokenized_train, max_sequence_length=ctrl_args.sequence_length
        )

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(train_args.output_dir) and train_args.do_train and not train_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(train_args.output_dir)
            if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({train_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and train_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        checkpoint = None
        if train_args.resume_from_checkpoint is not None:
            checkpoint = train_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        logger.debug(len(train_ds))
        logger.info("Number of params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            data_collator=collator
        )

        if train_args.do_train:
            trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()
            if is_kfold:
                torch.save(train_d, os.path.join(train_args.output_dir, 'fold.bin'))
                torch.save(ds.get_fold_ids(i_fold), os.path.join(train_args.output_dir, 'fold_ids.bin'))
                if hasattr(ds, 'bucket_sizes') and ds.bucket_sizes:
                    torch.save(ds.bucket_sizes, os.path.join(train_args.output_dir, "bucket_sizes.bin"))
            torch.save(init_emb, os.path.join(train_args.output_dir, CONTROL_CODES_INIT_FNAME))
    
    torch.save(ctrl_args, os.path.join(out_dir, 'ctrl_args.bin'))
    if is_kfold:
        ft_args.folds_shape = ds.folds.shape
    torch.save(ft_args, os.path.join(out_dir, 'ft_args.bin'))
    with open(os.path.join(out_dir, "control_codes.json"), 'w') as f:
        json.dump({
            "start": ds.get_control_code(),
            "end": ds.get_control_code(start=False)
        }, f)
