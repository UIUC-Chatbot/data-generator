import accelerate
from transformers import GPT2Tokenizer, OPTForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import datasets
from datasets import load_dataset, load_metric
import numpy as np

model = OPTForCausalLM.from_pretrained("facebook/opt-125m") # my favorite: facebook/opt-2.7b or 6.7b or 13b
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
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
custom_QA_dataset = load_dataset("json", data_files="./all_data.jl", field="gpt-3")
def tokenize_function(examples): return tokenizer(examples["text"], padding=True, truncation=True, max_length=741)
tokenized_datasets = custom_QA_dataset["train"].map(tokenize_function, batched=True)

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
    train_dataset=tokenized_datasets,
    # train_dataset=small_train_dataset,
    # eval_dataset=small_eval_dataset,
    # compute_metrics=compute_metrics,
)
trainer.train()
