from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from collections import OrderedDict
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

configuration = AutoConfig.from_pretrained('./')
student = AutoModelForSequenceClassification.from_config(configuration)
teacher = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-dianping-chinese',
                                                             output_hidden_states=True, return_dict=True)

temp = OrderedDict()
stu_state_dict = student.state_dict(destination=None)
for name, parameter in teacher.state_dict().items():
    for num in ['embeddings', '.0.', '.1.', '.2.', '.3.', '.4.', '.5.', '.6.', '.7.']:
        if num in name:
            temp[name] = parameter
stu_state_dict.update(temp)  # 更新参数值
student.load_state_dict(stu_state_dict)

training_args = TrainingArguments(output_dir="test_trainer",
                                  logging_dir="test_trainer/runs/small/8",
                                  evaluation_strategy="steps",
                                  eval_steps=200,
                                  num_train_epochs=3)


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
    model=student,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    #    optimizers=(optimizer, None)
)

trainer.train()
