import transformers
from datasets import load_dataset
from transformers import BertTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, AdamW
import math
from torch.optim import lr_scheduler

dataset = load_dataset("csv", data_files="dataset/train.csv")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="steps",
    eval_steps=200,
    num_train_epochs=10,
    # learning_rate=1e-4,
    # lr_scheduler_type="constant"
)

optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1200], gamma=0.1)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    optimizers=(optimizer, scheduler)
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
