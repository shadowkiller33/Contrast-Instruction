from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/llama-7b-reward-new",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model",
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
        default=1000,
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
    eval_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the human stack-exchange-paired dataset for tuning the reward model.

eval_dataset = load_dataset('json', data_files="/apdcephfs_cq2/share_1603164/data/lingfengshen/data/para/data_with_score.json", split='train')

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = script_args.model_name.split("/")[-1]
output_name = (
    f"{model_name_split}_peft_stack-exchange-paired_rmts__{script_args.train_subset}_{script_args.learning_rate}"
)

training_args = TrainingArguments(
    output_dir=output_name,
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
tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token


# model_name = "/home/lingfelshen/llama-7b-hf"
# adapters_name = "trl-lib/llama-7b-se-rm-peft"
#
# print(f"Starting to load the model {model_name} into memory")

model = AutoModelForSequenceClassification.from_pretrained(
    '/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/para-rm',
    torch_dtype=torch.float16,
    device_map={"": 0}
)
# model = PeftModel.from_pretrained(m, adapters_name)
# model = model.merge_and_unload()

# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = eval_dataset.column_names

human = []

# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
        "score_j": [],
        "score_k": [],
    }
    for query, response_j, response_k, score_j, score_k in zip(examples["question"], examples["response_j"], examples["response_k"],examples["response_j_score"], examples["response_k_score"]):

        tokenized_j = tokenizer("Paraphrase the following sentence: " + query + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Paraphrase the following sentence: " + query + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])
        new_examples["score_j"].append(score_j)
        new_examples["score_k"].append(score_k)


    return new_examples


eval_dataset = eval_dataset.select(range(4400,4900))

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)
for score_j, score_k in zip(eval_dataset["score_j"], eval_dataset["score_k"]):
    human.append(score_j)
    human.append(score_k)

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
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


reward = []
print(len(human))
class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=True):
        a = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])
        b = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])
        rewards_j = a[0]
        rewards_k = b[0]

        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        #print((rewards_j.cpu().detach().numpy(), rewards_k.cpu().detach().numpy()))
        reward.append(rewards_j.cpu().detach().numpy())
        reward.append(rewards_k.cpu().detach().numpy())
        # human.append(inputs["score_j"])
        # human.append(inputs["score_k"])
        #result.append((rewards_j.cpu().detach().numpy(), rewards_k.cpu().detach().numpy()))

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return 0


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    #compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)



trainer.predict(eval_dataset)

import pickle
with open("/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/reward_extro.pkl", "wb") as f:
    pickle.dump(reward, f)
with open('/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/human_extro.pkl','wb') as f:
    pickle.dump(human, f)
#
#
# import pickle
# with open('/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/para_contrast.pkl','rb') as f:
#     data = pickle.load(f)
# count = 0
# for x in result:
#     if x[0][0][0] > x[1][0][0]:
#         count += 1
# print(count/len(result))