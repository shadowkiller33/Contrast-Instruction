from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

import spacy

nlp = spacy.load('en_core_web_md')  # make sure to have this model downloaded
# find the synonym of the word using spacy
def get_synonyms(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    words = []
    for x in by_similarity:
        words.append(x.orth_)
    return words



DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(
        default="bert-base-uncased",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=50000,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
# accuracy = evaluate.load("accuracy")
# Load the human stack-exchange-paired dataset for tuning the reward model.
eval_dataset = load_dataset("/apdcephfs_cq2/share_1603164/data/lingfengshen/data/stack-exchange-paired", data_dir="data/evaluation", split="train")
if script_args.eval_subset > 0:
    eval_dataset = eval_dataset.select(range(script_args.eval_subset))



# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_auth_token=True)
config = AutoConfig.from_pretrained(script_args.model_name)

if "llama" in script_args.model_name:
    # required for llama
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
else:
    # required for gpt2
    tokenizer.pad_token = tokenizer.eos_token






# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token

num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = eval_dataset.column_names




model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.float16
)
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing


def process(examples):
    question, response_j, response_k = examples["question"], examples["response_j"], examples["response_k"]
    a_j = "Question: " + question + "\n\nAnswer: " + response_j
    a_k = "Question: " + question + "\n\nAnswer: " + response_k
    return a_j, a_k

def tokenize_(a_j, a_k):
    inputs = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    tokenized_j = tokenizer(a_j, truncation=True)
    tokenized_k = tokenizer(a_k, truncation=True)
    inputs["input_ids_j"].append(tokenized_j["input_ids"])
    inputs["attention_mask_j"].append(tokenized_j["attention_mask"])
    inputs["input_ids_k"].append(tokenized_k["input_ids"])
    inputs["attention_mask_k"].append(tokenized_k["attention_mask"])

    return inputs

def get_reward(a_j, a_k):
    inputs = tokenize_(a_j, a_k)
    rewards_j = model(input_ids=torch.Tensor(inputs["input_ids_j"]).to(torch.float16), attention_mask=torch.Tensor(inputs["attention_mask_j"]).to(torch.float16))[0]
    rewards_k = model(input_ids=torch.Tensor(inputs["input_ids_k"]).to(torch.float16), attention_mask=torch.Tensor(inputs["attention_mask_k"]).to(torch.float16))[0]
    return rewards_j - rewards_k

from simcse import SimCSE
sim_model = SimCSE("/apdcephfs_cq2/share_1603164/data/lingfengshen/models/sup-simcse-roberta-base")


import json
data = []
count = 0
for examples in eval_dataset:
    if count > 10:
        break
    case = {'question': examples["question"], 'response_j': examples["response_j"], 'response_k': examples["response_k"]}
    a_j, a_k = process(examples)
    reward_original = get_reward(a_j, a_k)

    flag = 0
    a_j_list = a_j.split()
    a_k_list = a_k.split()
    length_j = len(a_j_list)
    length_k = len(a_k_list)
    random_index_j = random.sample(range(length_j), int(0.2*length_j))
    random_index_k = random.sample(range(length_k), int(0.2*length_k))
    for (index_i,index_j) in zip(random_index_j,random_index_k):
        word_j = a_j_list[index_i]
        word_k = a_k_list[index_j]
        word_j_syn = get_synonyms(nlp.vocab[word_j])
        word_k_syn = get_synonyms(nlp.vocab[word_k])
        if word_j_syn != []:
            a_j_list[index_i] = word_j_syn[0]
        if word_k_syn != []:
            a_k_list[index_j] = word_k_syn[0]
        a_j_new = " ".join(a_j_list)
        a_k_new = " ".join(a_k_list)

        reward_new = get_reward(a_j_new, a_k_new)
        if reward_new < 0:
            flag = 1
            break
    if flag == 0:
        case['a_j_adv'] = 'failed'
        case['a_k_adv'] = 'failed'
        continue

    recover_sentence = []
    for (index_i, index_j) in zip(random_index_j, random_index_k):
        a_j_new_list = a_j_new.split()
        a_k_new_list = a_k_new.split()
        a_j_list = a_j.split()
        a_k_list = a_k.split()

        a_j_new_list[index_i] = a_j_list[index_i]
        a_k_new_list[index_j] = a_k_list[index_j]
        a_j_recovery = " ".join(a_j_new_list)
        a_k_recovery = " ".join(a_k_new_list)
        reward_recovery = get_reward(a_j_recovery, a_k_recovery)
        if reward_recovery < 0:
            similarity1 = sim_model.similarity(a_j_recovery, a_j)
            similarity2 = sim_model.similarity(a_k_recovery, a_k)
            recover_sentence.append((a_j_recovery, a_k_recovery, similarity1+similarity2))
    #find the tuple with the highest similarity
    recover_sentence.sort(key=lambda x:x[2], reverse=True)
    a_j_adv = recover_sentence[0][0]
    a_k_adv = recover_sentence[0][1]
    case['a_j_adv'] = a_j_adv
    case['a_k_adv'] = a_k_adv
    if reward_original > 0:
        case['label'] = 'True'
    else:
        case['label'] = 'False'
    data.append(case)
    count += 1

with open('/apdcephfs_cq2/share_1603164/data/lingfengshen/output/adv_output/data.json', 'w') as f:
    json.dump(data, f)


















