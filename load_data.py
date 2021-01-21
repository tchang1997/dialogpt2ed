from collections import defaultdict
from itertools import chain
import json
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer

import utils
from utils import PAD_VALUE

import logging
logger = logging.getLogger(__file__)


try:
    from datasets import load_dataset, list_datasets
except ImportError:
    logger.warning("Unable to import datasets (HuggingFace). Must use custom dataset spec.")


PAD_VALUE = -100
SPEAKER1_ID, SPEAKER2_ID = list(range(2))
# SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]

def hugging_face_load_dataset(name, split=None):
    return load_dataset(name, split=split)

def file_load_dataset(name, split=None):
    with open(name, "r") as f:
        if split:
            return json.load(f)[split]
        else:
            return json.load(f)

class HuggingFaceDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.config = data_config
        for key in self.config:
            setattr(self, key, self.config[key])

    def setup(self, stage):
        logger.info("Setting up tokenizer and raw data...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.name in list_datasets():
            logger.info("Loading HuggingFace dataset...")
            self.dataset_setup_fn = hugging_face_load_dataset
        else:
            logger.info("Loading local dataset...")
            self.dataset_setup_fn = file_load_dataset
            if not os.path.isfile(self.name):
                raise FileNotFoundError(f"Passed in path `{self.name}` for dataset, but no such file found.")
        if stage == 'train':
            self.train = self.dataset_setup_fn(self.name, split="train")
            self.val = self.dataset_setup_fn(self.name, split="valid")
        elif stage == 'test':
            # DSYITF - don't  shoot yourself in the foot. Comment this out when doing pre-prod testing. 
            self.val = self.dataset_setup_fn(self.name, split="valid")
            # self.test = self.dataset_setup_fn(self.name, split="test")
        else:
            raise NotImplementedError()


    def attach_special_tokens(self, model):
        orig_num_tokens = self.tokenizer.vocab_size
        num_added_tokens = self.tokenizer.add_special_tokens(utils.ATTR_TO_SPECIAL_TOKEN)
        if num_added_tokens > 0:
            model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


    def load_tokenized_dataset(self, dataset, cache):

        def tokenize(obj):
            if isinstance(obj, str):
                return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        if cache and os.path.isfile(cache):
            logger.info("Load tokenized dataset from cache at %s", cache)
            dataset = torch.load(cache)
        else:
            logger.info("Tokenize and encode the dataset for cache {}".format(cache))
            dataset = tokenize(dataset)
            logger.info('Finished tokenizing, saving to {}'.format(cache))
            torch.save(dataset, cache)
        return dataset

    def build_inputs_and_labels(self, dataset):
        dataset_info = defaultdict(list)
        num_candidates = len(dataset[0]["turns"][0]["candidates"]) # hack?
        if self.num_candidates > 0:
            num_candidates = min(self.num_candidates, num_candidates)
        for dialog in dataset:
            for utterance in dialog["turns"]:
                history = utterance["history"][-(2*self.max_history+1):] # get current half-turn + last self.max_history turns
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)  # only get the lm target when we're using the last candidate (i.e. the gold candidate)
                    instance = self.build_input_from_segments(history, candidate, lm_labels=lm_labels)
                    for input_name, input_array in instance.items():
                        dataset_info[input_name].append(input_array)
                dataset_info["mc_labels"].append(num_candidates - 1)  # the index of the correct answer for the multiple choice
                dataset_info["n_candidates"] = num_candidates
        return dataset_info


    def build_input_from_segments(self, history, reply, lm_labels=False, with_eos=True):
        # bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
        instance = {}
        sequence = history + [reply + ([self.tokenizer.eos_token_id] if with_eos else [])]
        # sequence = [([bos] if i==0 else []) + [speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence)]

        instance["input_ids"] = list(chain(*sequence))  # list of ints
        instance["token_type_ids"] = [SPEAKER2_ID if i % 2 else SPEAKER1_ID for i, s in enumerate(sequence) for _ in s]  # list of ints (all speaker1 or speaker2, starting with speaker1), same length as input_ids
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1  # int, the length of the whole input. it gives the location of the last hidden state, from which we compute the multiple choice loss
        instance["labels"] = [PAD_VALUE] * len(instance["input_ids"])  # -100 for the whole sequence if lm_labels=False
        if lm_labels:
            instance["labels"] = ([PAD_VALUE] * sum(len(s) for s in sequence[:-1])) + [PAD_VALUE] + sequence[-1][1:]  # -1 for the masked parts, then the actual targets for the reply
        return instance


    def pad_and_encode(self, dataset):
        tensor_dataset = []
        dataset = utils.pad_dataset(dataset,
                padding=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
        for input_name in utils.MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, dataset["n_candidates"]) + tensor.shape[1:])
            tensor_dataset.append(tensor)
        return tensor_dataset

    def featurize(self, dataset, cache=None):
        tokenized = self.load_tokenized_dataset(dataset, cache)
        restructured = self.build_inputs_and_labels(tokenized)
        tensorified = self.pad_and_encode(restructured)
        return TensorDataset(*tensorified)

    def train_dataloader(self):
        with utils.TimerContext("Loading train dataloader"):
            train_processed = self.featurize(self.train, cache=f"{os.path.splitext(self.dataset_cache)[0]}_train.bin")
        train_dl = DataLoader(train_processed, num_workers=os.cpu_count(), batch_size=self.batch_size)
        return train_dl

    def val_dataloader(self):
        with utils.TimerContext("Loading validation dataloader"):
            val_processed = self.featurize(self.val, cache=f"{os.path.splitext(self.dataset_cache)[0]}_valid.bin")
        val_dl = DataLoader(val_processed, num_workers=os.cpu_count(), batch_size=self.batch_size)
        return val_dl

    def test_dataloader(self):
        with utils.TimerContext("loading testing dataloader"):
            logger.warn("You have loaded the testing dataloader. Please ensure that you did this on purpose!")
            test_processed = self.featurize(self.test, cache=f"{os.path.splitext(self.dataset_cache)[0]}_test.bin")
        test_dl = DataLoader(test_processed, num_workers=os.cpu_count(), batch_size=self.batch_size)
        return test_dl


