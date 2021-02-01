import pytorch_lightning as pl
import transformers
from transformers import GPT2DoubleHeadsModel
import torch
import torch.optim as optim

import logging
logger = logging.getLogger(__file__)

try:
  import wandb
except ImportError:
  logger.warning("Unable to import wandb. Table-level logging will not work -- only inference, or training with no logging will work")
from pytorch_lightning.metrics import Accuracy

from load_data import SPEAKER1_ID
from utils import MODEL_INPUTS, PAD_VALUE

class HuggingFaceModel(pl.LightningModule):

    def __init__(self, model_name, config, do_fine_tune=False):
        super().__init__()

        # todo: validate config structure
        self.config = config
        self.model_name = model_name
        if do_fine_tune:
            self.model = GPT2DoubleHeadsModel.from_pretrained(model_name)
        self.curr_eval_table = []
        self.accuracy = Accuracy()

    def configure_optimizers(self):
        opt_config = self.config["optimizer"]
        opt_name = opt_config["name"]
        if hasattr(optim, opt_config["name"]):
            try: # Default: PyTorch optimizer
                optimizer = getattr(optim, opt_name)(self.model.parameters(), **opt_config["kwargs"]) # must include LR, for one           
            except TypeError: # possibly a transformers optimizer (AdamW)
                optimizer = getattr(transformers, opt_name)(self.model.parameters(), **opt_config["kwargs"])
        else:
            raise Exception('Unexpected learning algorithm "{}"'.format(opt_name))
        return optimizer

    def attach_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, batch):
        batch[1] = batch[1].squeeze(-1) # mc_token_ids
        batch[3] = batch[3].squeeze(-1) # mc_labels
        inputs = dict(zip(MODEL_INPUTS, batch))
        return self.model(**inputs)

    
    def training_step(self, batch, batch_idx):
        # model type: GPT2LMHEadModel (https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel)
        train_config = self.config["train"]
        outputs = self(batch)
        lm_loss, mc_loss, _, mc_logits = outputs[:4]
        loss = lm_loss * train_config["lm_weight"] + mc_loss * train_config["mc_weight"]
        mc_acc = self.accuracy(mc_logits, batch[MODEL_INPUTS.index("mc_labels")])
        self.log('loss', loss)
        self.log('mc_acc', mc_acc, prog_bar=True)
        self.log('lm_loss', lm_loss, prog_bar=True)
        self.log('mc_loss', mc_loss, prog_bar=True)
        return {'loss': loss}

    def eval_step(self, batch, batch_idx):
        train_config = self.config["train"]
        outputs = self(batch)
        lm_loss, mc_loss, _, mc_logits = outputs[:4]
        loss = lm_loss * train_config["lm_weight"] + mc_loss * train_config["mc_weight"]

        mc_labels = batch[MODEL_INPUTS.index("mc_labels")]
        mc_acc = self.accuracy(mc_logits, mc_labels)
    
        input_ids = batch[MODEL_INPUTS.index("input_ids")] # (bs, 2, len)
        distractor, orig = input_ids[:, 0], input_ids[:, 1]

        orig_token_type_ids = batch[MODEL_INPUTS.index("token_type_ids")][:, 1]
        targets = torch.index_select(batch[MODEL_INPUTS.index("labels")], 1, mc_labels.view(-1)).squeeze(1) # (bs, len)
        short_distractor = distractor[orig != distractor] # (bs, short_len)
        eos_tensor = torch.tensor([self.tokenizer.eos_token_id], device=self.model.device)
        short_orig = torch.cat([orig[orig_token_type_ids == SPEAKER1_ID], eos_tensor], dim=-1) # [any, any, any, ... , <eos>]
        if short_orig.ndim == 1: short_orig = short_orig.unsqueeze(0) 
        candidate_sents = self.model.generate(short_orig,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.config['inference'])
        self.log_text_predictions(short_orig,
                short_distractor,
                targets,
                candidate_sents[:, short_orig.size(-1):])
        self.log('val_loss', loss)
        self.log('val_mc_acc', mc_acc)
        self.log('val_lm_loss', lm_loss)
        self.log('val_mc_loss', mc_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def eval_epoch_end(self, batches, table_name):
        table = wandb.Table(data=self.curr_eval_table,
                        columns=["Original", "Target", "Distractor", "Predicted"])
        self.logger.experiment.log({table_name: table})
        self.curr_eval_table = []

    def clean_text_from_tensor(self, ids):
        raw_tokens = self.tokenizer.convert_ids_to_tokens(ids.view(-1).tolist())
        clean_tokens = [token.replace("<pad>", "").replace(chr(288), "") for token in raw_tokens]
        return " ".join(clean_tokens)

    def log_text_predictions(self, orig, distractor, labels, predictions):
        original_text = self.tokenizer.batch_decode(orig, skip_special_tokens=True)
        predictions_text = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = labels[labels != PAD_VALUE]
        # handle batch size = 1 edge case
        if labels.ndim == 1: labels = labels.unsqueeze(0)
        if distractor.ndim == 1: distractor = distractor.unsqueeze(0) 
        targets_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        distractor_text = self.tokenizer.batch_decode(distractor, skip_special_tokens=True)
        self.curr_eval_table += list(zip(original_text, targets_text, distractor_text, predictions_text))

    def validation_epoch_end(self, batches):
        self.eval_epoch_end(batches, f"textgen_val_{self.current_epoch}_step{self.global_step}")

    def test_epoch_end(self, batches):
        self.eval_epoch_end(batches, "textgen_test")
