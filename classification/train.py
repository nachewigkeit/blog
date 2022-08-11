from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
metric = load_metric("accuracy")


def tokenize_function(examples):
    return tokenizer(examples["review"], padding='max_length', truncation=True, max_length=274)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


dataset = load_dataset("csv", data_files={"train": "dataset/train.csv", "test": "dataset/test.csv"})
tokenized_datasets = dataset.map(tokenize_function, batched=True)

unfreeze_layers = []

model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-dianping-chinese',
                                                           num_labels=2)
training_args = TrainingArguments(output_dir="test_trainer",
                                  logging_dir="test_trainer/runs/class_weight/",
                                  evaluation_strategy="steps",
                                  eval_steps=200,
                                  num_train_epochs=3)


# unfreeze_layers.append(str(i))
# for name, param in model.named_parameters():
#     param.requires_grad = False
#     for ele in unfreeze_layers:
#         if ele in name:
#             param.requires_grad = True
#             break

# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=6e-5)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]).cuda())
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    #    optimizers=(optimizer, None)
)

trainer.train()
