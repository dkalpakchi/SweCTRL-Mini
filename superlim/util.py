import numpy as np
from numpy.linalg import lstsq
from scipy.linalg import orth
import dataclasses as dc

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

from common import UNKNOWN_TOKEN, PAD_TOKEN
from control_codes import START_C_CODES, END_C_CODES


@dc.dataclass
class CtrlArguments:
    train_data: str = dc.field(
        default="data/training_cunique_with_distractors.json",
        metadata={"help": "A CSV list of training data files"}
    )

    formulation: str = dc.field(
        default="areg_ltr",
        metadata={"help": "Type of problem definition: autoregressive (areg) or u-PMLM (upmlm) or mixed (if predict_questions is set)"}
    )

    context_strategy: str = dc.field(
        default="take_first",
        metadata={"help": "How to deal with contexts greater than a specified length"}
    )

    tokenizer_file: str = dc.field(
        default="tokenizer.json",
        metadata={"help": "A JSON file (in the format provided by HuggingFace's tokenizers library) with a trained tokenizer"}
    )

    sequence_length: int = dc.field(
        default=256,
        metadata={"help": "The max sequence length"}
    )

    force_prepend_control: bool = dc.field(
        default=False,
        metadata={"help": "If the control code should be prepended for all sliding windows. Otherwise, it is only prepended at the start of the sequence"}
    )


@dc.dataclass
class FineTuningArguments:
    superlim_dataset: str = dc.field(
        metadata={"help": "A name of a SuperLim dataset"}
    )

    ctrl_checkpoint: str = dc.field(
        metadata={"help": "A path to the file with the desired CTRL checkpoint"}
    )
    
    ctrl_args_file: str = dc.field(
        metadata={"help": "A path to the file with the arguments of the trained CTRL model"}
    )
    
    n_folds: int = dc.field(
        default=20,
        metadata={"help": "Number of folds"}
    )
    
    fold_size: int = dc.field(
        default=50,
        metadata={"help": "The number of datapoints in a fold"}
    )


class GradientPrinter:
    def __init__(self, name):
        self.name = name

    def __call__(self, grad):
        np_grad = grad.cpu().numpy()
        print("======== GRAD FOR {} ========".format(self.name))
        print("\tGRAD {}".format(grad))
        print("\tGRAD NORM {}".format(np.linalg.norm(np_grad)))
        print("\tGRAD MEAN {}".format(np.mean(np_grad)))
        print()


def load_tokenizer(fname, additional_start_tokens, additional_end_tokens):
    tok = PreTrainedTokenizerFast(tokenizer_file=fname)
    tok.pad_token = PAD_TOKEN
    tok.unk_token = UNKNOWN_TOKEN

    res = tok.add_special_tokens({
        'additional_special_tokens': list(additional_start_tokens.values()) +\
                list(additional_end_tokens.values())
    })

    START_C_CODES.update(additional_start_tokens)
    END_C_CODES.update(additional_end_tokens)
    return tok, START_C_CODES, END_C_CODES


def reassign_embeddings(model, new_emb):
    # Build new embeddings
    new_embeddings = nn.Embedding(*new_emb.shape).to(model.base_model.device)
    new_embeddings.weight.data = torch.tensor(new_emb).float()
    model.transformer.set_input_embeddings(new_embeddings)


def find_orth(O):
    # Taken from: https://stackoverflow.com/questions/50660389/generate-a-vector-that-is-orthogonal-to-a-set-of-other-vectors-in-any-dimension
    rand_vec = np.random.rand(O.shape[0], 1)
    A = np.hstack((O, rand_vec))
    b = np.zeros(O.shape[1] + 1)
    b[-1] = 1
    return lstsq(A.T, b)[0]
