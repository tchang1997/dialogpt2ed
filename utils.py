from contextlib import AbstractContextManager
import io
import sys
import time
import torch

from typing import Optional


ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                                 'additional_special_tokens': ('<speaker1>', '<speaker2>')}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "labels", "mc_labels", "token_type_ids"]  # labels = lm_labels
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]
LOGGER_FORMAT = "[%(levelname)s] %(name)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
PAD_VALUE = -100

def pad_dataset(pad_token, batch):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    input_ids_lens = [example[MODEL_INPUTS.index("input_ids")].size(-1) for example in batch]
    max_l = max(input_ids_lens)
    for i in range(len(batch)): # TODO: this is not very efficient.  make it efficient.
        curr_example = batch[i]
        for name in PADDED_INPUTS:
            curr_data = curr_example[MODEL_INPUTS.index(name)]
            seq_len = curr_data.size(-1)
            batch[i][MODEL_INPUTS.index(name)] = torch.cat([curr_data, (pad_token if name != "labels" else PAD_VALUE) * torch.ones(max_l - seq_len)], dim=-1)
    inputs = list(zip(*batch))
    inputs = [torch.stack(tensor_tuple, dim=0).long() for tensor_tuple in inputs]
    return inputs


class TimerContext(AbstractContextManager):
    def __init__(self, message: str, prefix: Optional[str] = ">", 
                 file: Optional[io.TextIOWrapper] = sys.stdout,
                 precision: Optional[int] = 4):
        self.file = file
        self.prefix = prefix
        self.message = message
        self.fmt_string = f"{{:.{precision}f}}"

    def __enter__(self) -> "timer_context":
        self.start = time.time()
        return self

    def __exit__(self, *exc) -> bool:
        elapsed = time.time() - self.start
        print(self.prefix, self.message, f"took {self.fmt_string}s".format(elapsed),
              file=self.file)
        return False
