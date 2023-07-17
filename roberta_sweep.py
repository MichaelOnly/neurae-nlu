import wandb
import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

def preprocess_function(examples):
    return tokenizer(examples['examples'], truncation=True, return_tensors='pt', padding='max_length')

dataset = load_dataset("neurae/dnd_style_intents")

train_dataset = dataset["train"]
eval_dataset = dataset["eval"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    metrics = dict()
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    preds = np.argmax(logits, axis=-1)
    metrics = {
        'accuracy': accuracy.compute(predictions=preds, references=labels),
        'precision-micro': precision.compute(predictions=preds, references=labels, average='micro'),
        'precision-macro': precision.compute(predictions=preds, references=labels, average='macro'),
        'recall-micro': recall.compute(predictions=preds, references=labels, average='micro'),
        'recall-macro': recall.compute(predictions=preds, references=labels, average='macro'),
        'f1-micro': f1.compute(predictions=preds, references=labels, average='micro'),
        'f1-macro': f1.compute(predictions=preds, references=labels, average='macro')
    }
    return metrics

model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion", num_labels=17, ignore_mismatched_sizes=True)

train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
eval_tokenized_dataset = eval_dataset.map(preprocess_function, batched=True)

train_tokenized_dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)
eval_tokenized_dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

sweep_config = {
   'method': 'bayes',
   'name': 'sweep',
   'metric': {
      'goal': 'minimize', 
      'name': 'eval_loss'
      },
   'parameters': {
      'lr': {'max': 1e-3, 'min': 1e-5},
      'scheduler_type': { 'values': ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']},
      'weight_decay': {'min': 0.0, 'max': 0.1}
   }
}
sweep_id = wandb.sweep(
      sweep_config, project="roberta-sweep")

def train_with_wandb(config=None):
   with wandb.init(config=config):
      config = wandb.config
      training_args = TrainingArguments(
            evaluation_strategy="epoch",
            output_dir="models/roberta",
            overwrite_output_dir=True,
            logging_strategy="steps",
            logging_dir="models/roberta/logs",
            logging_steps=100,
            learning_rate=config.lr,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=8,
            weight_decay=config.weight_decay,
            report_to="wandb",
            load_best_model_at_end=True,
            save_strategy="epoch",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            torch_compile=True,
            bf16=True,
            optim="adafactor",
            lr_scheduler_type=config.scheduler_type,
            gradient_accumulation_steps=4
      )

      trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized_dataset,
            eval_dataset=eval_tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
      )

      trainer.train()

wandb.agent(sweep_id, train_with_wandb, count=10)