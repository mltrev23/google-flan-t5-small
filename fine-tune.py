from datasets import load_dataset

dataset = load_dataset("code_x_glue_ct_code_to_text", "ruby")
print(dataset)

example = dataset['train'][0]

print("Code:", example["code"])
print("Docstring:", example["docstring"])

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")

prefix = "Summarize Ruby: "
max_input_length = 256
max_target_length = 128

def preprocess_examples(examples):
  # encode the code-docstring pairs
  codes = examples['code']
  docstrings = examples['docstring']
  
  inputs = [prefix + code for code in codes]
  model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

  # encode the summaries
  labels = tokenizer(docstrings, max_length=max_target_length, padding="max_length", truncation=True).input_ids

  # important: we need to replace the index of the padding tokens by -100
  # such that they are not taken into account by the CrossEntropyLoss
  labels_with_ignore_index = []
  for labels_example in labels:
    labels_example = [label if label != 0 else -100 for label in labels_example]
    labels_with_ignore_index.append(labels_example)
  
  model_inputs["labels"] = labels_with_ignore_index

  return model_inputs

dataset = dataset.map(preprocess_examples, batched=True)
#print(dataset)

from torch.utils.data import DataLoader

dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=8)
valid_dataloader = DataLoader(dataset['validation'], batch_size=4)
test_dataloader = DataLoader(dataset['test'], batch_size=4)

batch = next(iter(train_dataloader))
print(batch.keys())
"""
#all
out = [tokenizer.decode(outtoken) for outtoken in batch['input_ids']]
print(out)
print('-------------------------')
#labels = batch['labels']
out = [tokenizer.decode([label for label in labels if label != -100]) for labels in batch['labels']]
print(out)
"""

out = tokenizer.decode(batch['input_ids'][0]) 
print(out)
print('-------------------------')
labels = batch['labels'][0]
out = tokenizer.decode([label for label in labels if label != -100])
print(out)


from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl

class CodeT5(pl.LightningModule):
    def __init__(self, lr=5e-5, num_train_epochs=15, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss
      
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader
"""
import wandb

wandb.login()
model = CodeT5()

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

wandb_logger = WandbLogger(name='codet5-finetune-code-summarization-ruby-shuffle', project='CodeT5')
# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = Trainer(gpus=1, 
                  default_root_dir="/content/drive/MyDrive/CodeT5/Notebooks/Checkpoints", 
                  logger=wandb_logger, 
                  callbacks=[early_stop_callback, lr_monitor])
trainer.fit(model)

save_directory = "." # save in the current working directory, you can change this of course
model.model.save_pretrained(save_directory)

from datasets import load_dataset

dataset = load_dataset("code_x_glue_ct_code_to_text", "ruby")
print(dataset['test'])

test_example = dataset['test'][2]
print("Code:", test_example['code'])

from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(save_directory)

# prepare for the model
input_ids = tokenizer(test_example['code'], return_tensors='pt').input_ids
# generate
outputs = model.generate(input_ids)
print("Generated docstring:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("Ground truth:", test_example['docstring'])
"""