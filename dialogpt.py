import pytorch_lightning as pl
import transformers
from transformers import GPT2DoubleHeadsModel
import torch.optim as optim
import wandb

from utils import MODEL_INPUTS

class HuggingFaceModel(pl.LightningModule):

    def __init__(self, model_name, config):
        super().__init__()

        # todo: validate config structure
        self.config = config
        self.model_name = model_name
        self.model = GPT2DoubleHeadsModel.from_pretrained(model_name)
        self.curr_eval_table = []

    def configure_optimizers(self):
        opt_config = self.config["optimizer"]
        if hasattr(optim, opt_config["name"]):
            try: # Default: PyTorch optimizer
                optimizer = getattr(optim, opt_config["name"])(self.model.parameters(), **opt_config["kwargs"]) # must include LR, for one           
            except TypeError: # possibly a transformers optimizer (AdamW)
                optimizer = getattr(transformers, opt_config["name"])(self.model.parameters(), **opt_config["kwargs"])
        else:
            raise Exception('Unexpected learning algorithm "{}"'.format(learning_alg))

    def forward(self, batch):
        inputs = dict(zip(MODEL_INPUTS, batch))
        return self.model(**inputs)

    # todo: nuke this
    def training_step(self, batch, batch_idx):
        # model type: GPT2LMHEadModel (https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel)
        train_config = self.config["train"]
        lm_loss, mc_loss, *_ = self(batch)
        loss = lm_loss * train_config["lm_weight"] + mc_loss * train_config["args.mc_weight"]
        self.log('val_loss', loss)
        self.log('lm_loss', lm_loss)
        self.log('mc_loss', mc_loss)
        return loss

    def eval_step(self, batch, batch_idx):
        lm_loss, mc_loss, *_ = self(batch)
        loss = lm_loss * train_config["lm_weight"] + mc_loss * train_config["args.mc_weight"]
        candidate_sents = self.model.generate(**self.config['inference'])
        self.log_text_predictions(batch["input_ids"], batch["lm_labels"], candidate_sents)
        self.log('val_loss', loss)
        self.log('lm_loss', lm_loss)
        self.log('mc_loss', mc_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def eval_epoch_end(self, batches, table_name):
        table = wandb.Table(data=self.curr_eval_table,
                        columns=["Original", "Target", "Predicted"])
        self.logger.experiment.log({table_name: table})
        self.curr_eval_table = []

    def log_text_predictions(self, originals, targets, predictions):
        original_text = self.tokenizer.convert_ids_to_tokens(originals.view(-1).tolist(), skip_special_tokens=True)
        original_text = self.tokenizer.convert_tokens_to_string(original_text)
        unpadded_targets = targets[targets != -1] # todo: not hardcoded
        predictions_text = self.tokenizer.convert_ids_to_tokens(predictions.view(-1).tolist(), skip_special_tokens=True)
        predictions_text = self.tokenizer.convert_tokens_to_string(predictions_text)
        targets_text = self.tokenizer.convert_ids_to_tokens(targets.view(-1).tolist(), skip_special_tokens=True)
        targets_text = self.tokenizer.convert_tokens_to_string(targets_text)
        self.curr_eval_table += list(zip(original_text, predictions_text, targets_text))

    def validation_epoch_end(self, batches):
        self.eval_epoch_end(batches, f"textgen_val_{self.current_epoch}_step{self.global_step}")

    def test_epoch_end(self, batches):
        self.eval_epoch_end(batches, "textgen_test")
