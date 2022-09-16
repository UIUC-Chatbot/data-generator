import accelerate
from transformers import GPT2Tokenizer, OPTForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import datasets
from datasets import load_dataset, load_metric
import numpy as np
from itertools import chain


model = OPTForCausalLM.from_pretrained("facebook/opt-125m") # my favorite: facebook/opt-2.7b or 6.7b or 13b
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
tokenizer.add_bos_token = False

model.train()
max_mem = 1024 * 1024 * 22_000
# max_mem = 1800010241024
device_map = accelerate.infer_auto_device_map(
    model, 
    max_memory={0: max_mem,1:max_mem,2:max_mem,3:max_mem},
    no_split_module_classes=["OPTDecoderLayer"], 
    dtype='float16' # todo: bfloat16
)
model = accelerate.dispatch_model(model, device_map=device_map, offload_dir='./hf_offload_dir', offload_buffers=False)

# load & tokenize custom dataset
datasets.disable_caching() # Don't use dataset cache (for debugging)
# todo: figure out what the forward pass needs https://huggingface.co/docs/transformers/model_doc/opt#transformers.OPTForCausalLM.forward
# todo: to help with above, find tokenizer params here: https://huggingface.co/docs/transformers/v4.21.3/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
# def tokenize_function(examples): return tokenizer.encode(examples["text"], padding=True, truncation=True, max_length=2048, return_tensors="pt")
def tokenize_function(examples): return tokenizer(examples["text"])
custom_QA_dataset = load_dataset("json", data_files="./all_data.jl", field="gpt-3")
tokenized_datasets = custom_QA_dataset["train"].map(tokenize_function, batched=True, remove_columns=custom_QA_dataset["train"].column_names)

block_size = 1024
# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )

print(tokenized_datasets)

# TODO: use built-in metric comparison. Just need to add a train/test split.
# metric = load_metric("accuracy")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(20))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(20))
# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# todo try: auto_find_batch_size=True
# prediction_loss_only
training_args = TrainingArguments(output_dir="opt_fine_tune_logs", per_device_train_batch_size=1, prediction_loss_only=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    # train_dataset=small_train_dataset,
    # eval_dataset=small_eval_dataset,
    # compute_metrics=compute_metrics,
)
trainer.train()
