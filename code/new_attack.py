from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import pandas as pd
import evaluate
import numpy as np
import random
import torch.nn as nn
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
import datasets
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

import spacy

nlp = spacy.load('/apdcephfs_cq2/share_1603164/data/lingfengshen/models/en_core_web_md/en_core_web_md-3.5.0')  # make sure to have this model downloaded
# find the synonym of the word using spacy
def get_synonyms(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    words = []
    for x in by_similarity:
        words.append(x.orth_)
    return words[-1]

from simcse import SimCSE
sim_model = SimCSE("/apdcephfs_cq2/share_1603164/data/lingfengshen/models/sup-simcse-roberta-base")



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
        default="gpt2",
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

eval_dataset = load_dataset("/apdcephfs_cq2/share_1603164/data/lingfengshen/data/stack-exchange-paired", data_dir="data/evaluation", split="train")
if script_args.eval_subset > 0:
    eval_dataset = eval_dataset.select(range(script_args.eval_subset))
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = script_args.model_name.split("/")[-1]


training_args = TrainingArguments(
    output_dir='output_name',
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
)
# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, truncation=True, max_length=256, use_auth_token=True)
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



model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1#, torch_dtype=torch.float16
)



# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = eval_dataset.column_names


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


# preprocess the dataset and filter out QAs that are longer than script_args.max_length





# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 256
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("/apdcephfs_cq2/share_1603164/data/lingfengshen/eval/accuracy")

def process(examples):
    question, response_j, response_k = examples["question"], examples["response_j"], examples["response_k"]
    a_j = "Question: " + question + "\n\nAnswer: " + response_j
    a_k = "Question: " + question + "\n\nAnswer: " + response_k
    return a_j, a_k
def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=True):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
json_data = []

N = 2000
for i in tqdm(range(N)):
    data = eval_dataset[i]
    data_slice = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    eval_slice = data_slice.map(preprocess_function, batched=True, remove_columns=original_columns)
    # eval_slice = eval_slice.filter(
    #     lambda x: len(x["input_ids_j"]) <= 512 and len(x["input_ids_k"]) <= 512
    # )
    # if len(eval_slice) == 0:
    #     continue

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=eval_slice,
        eval_dataset=eval_slice,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    )

    result = trainer.predict(eval_slice)
    # print(result)
    # print(type(result))
    print("Evaluating on slice " + str(i) + ".")
    acc = result[2]["test_accuracy"]

    reward_j = result[0][0][0][0]
    reward_k = result[0][1][0][0]
    # print(reward_k)
    # print(reward_j)
    case = {'id': int(i),
            'question': data["question"],
            'response_j': data["response_j"],
            'response_k': data["response_k"],
            'reward_j': [],
            'reward_k': [],
            'reward_query_j': [],
            'reward_query_k': [],
            'index_j': [],
            'index_k': []}
    if acc == 1:
        case['label'] = 'True'
    else:
        case['label'] = 'False'
    case['reward_j'].append(reward_j)
    case['reward_k'].append(reward_k)
    a_j, a_k = process(data)

    flag = 0
    a_j_list = a_j.split()
    a_k_list = a_k.split()
    length_j = len(a_j_list)
    length_k = len(a_k_list)
    random_index_j = random.sample(range(length_j), int(0.3*length_j))
    random_index_k = random.sample(range(length_k), int(0.3*length_k))


    candidate_j = a_j_list
    candidate_k = a_k_list
    count1 = 0
    count2 = 0
    query_j = 0
    query_k = 0
    for (index_i,index_j) in zip(random_index_j,random_index_k):
        s_j = candidate_j
        s_k = candidate_k

        word_j = candidate_j[index_i]
        word_k = candidate_k[index_j]
        word_j_syn = get_synonyms(nlp.vocab[word_j])
        word_k_syn = get_synonyms(nlp.vocab[word_k])
        if word_j_syn == '' and word_k_syn == '':
            continue

        s_j[index_i] = word_j_syn
        s_k[index_j] = word_k_syn

        a_j_new = " ".join(s_j)
        a_k_new = " ".join(s_k)
        data_new = eval_dataset[i]
        data_new["response_j"] = a_j_new
        data_new["response_k"] = a_k_new
        data_slice_new = datasets.Dataset.from_pandas(pd.DataFrame(data=data_new))
        eval_slice_new = data_slice_new.map(preprocess_function, batched=True, remove_columns=original_columns)
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=eval_slice_new,
            eval_dataset=eval_slice_new,
            compute_metrics=compute_metrics,
            data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
        )
        query_j += 1
        query_k += 1
        result = trainer.predict(eval_slice_new)
        reward_j_new = result[0][0][0][0]
        reward_k_new = result[0][1][0][0]
        if reward_j_new < reward_j:
            reward_j = reward_j_new
            candidate_j = s_j
            case['reward_query_j'].append(reward_j_new)
            count1 += 1

        if reward_k_new > reward_k:
            reward_k = reward_k_new
            candidate_k = s_k
            case['reward_query_k'].append(reward_k_new)
            count2 += 1

        case['reward_j'].append(reward_j_new)
        case['reward_k'].append(reward_k_new)
        case['index_j'].append(count1/length_j)
        case['index_k'].append(count2/length_k)

        acc_new = reward_j_new - reward_k_new#result[2]["test_accuracy"]
        if acc_new <= 0:
            case['count_j'] = count1
            case['count_k'] = count2
            case['query_j'] = query_j
            case['query_k'] = query_k
            case['a_j_adv'] = " ".join(candidate_j)
            case['a_k_adv'] = " ".join(candidate_j)
            json_data.append(case)
            flag = 1
            break
    if flag == 0:
        case['count_j'] = count1
        case['count_k'] = count2
        case['a_j_adv'] = 'failed'
        case['a_k_adv'] = 'failed'
        json_data.append(case)
        continue






    # recover_sentence = []
    # for (index_i, index_j) in zip(random_index_j, random_index_k):
    #     a_j_new_list = candidate_j
    #     a_k_new_list = candidate_k
    #     a_j_list = a_j.split()
    #     a_k_list = a_k.split()
    #
    #     a_j_new_list[index_i] = a_j_list[index_i]
    #     a_k_new_list[index_j] = a_k_list[index_j]
    #     a_j_recovery = " ".join(a_j_new_list)
    #     a_k_recovery = " ".join(a_k_new_list)
    #     data_recovery = eval_dataset[i]
    #     data_recovery["response_j"] = a_j_recovery
    #     data_recovery["response_k"] = a_k_recovery
    #     data_slice_recovery = datasets.Dataset.from_pandas(pd.DataFrame(data=data_recovery))
    #     eval_slice_recovery = data_slice_recovery.map(preprocess_function, batched=True, remove_columns=original_columns)
    #     trainer = RewardTrainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=eval_slice_recovery,
    #         eval_dataset=eval_slice_recovery,
    #         compute_metrics=compute_metrics,
    #         data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    #     )
    #
    #     result = trainer.predict(eval_slice_recovery)
    #     reward_j_new = result[0][0][0][0]
    #     reward_k_new = result[0][1][0][0]
    #     if reward_j_new <= reward_k_new:
    #         similarity1 = sim_model.similarity(a_j_recovery, a_j)
    #         similarity2 = sim_model.similarity(a_k_recovery, a_k)
    #         recover_sentence.append((a_j_recovery, a_k_recovery, similarity1+similarity2))
    #find the tuple with the highest similarity
    # recover_sentence.sort(key=lambda x:x[2], reverse=True)
    # a_j_adv = recover_sentence[0][0]
    # a_k_adv = recover_sentence[0][1]
    # case['a_j_adv'] = a_j_adv
    # case['a_k_adv'] = a_k_adv
    # json_data.append(case)
import pickle
with open('/apdcephfs_cq2/share_1603164/data/lingfengshen/output/adv_output/wmt' + str(N) + '_data.pkl', 'wb') as f:
    pickle.dump(json_data, f)


