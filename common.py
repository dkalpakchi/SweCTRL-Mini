from transformers import PreTrainedTokenizerFast

UNKNOWN_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


def init_tokenizer(fname):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=fname)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.unk_token = UNKNOWN_TOKEN
    return tokenizer
