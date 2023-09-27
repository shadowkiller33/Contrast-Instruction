from transformers import pipeline, AutoTokenizer
import pickle
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import torch
from datasets import load_dataset
train_dataset = load_dataset("/apdcephfs_cq2/share_1603164/data/lingfengshen/data/stack-exchange-paired", data_dir="data/rl", split="train").select(range(100))

tokenizer = AutoTokenizer.from_pretrained('/apdcephfs_cq2/share_1603164/data/huggingface_models/llama-7b-hf')
tokenizer.add_special_tokens(
    {
        "eos_token": "</s>",
        "bos_token": "</s>",
        "unk_token": "</s>",
    }
)
# with open('ml_content.pkl', 'rb') as f:
#     data = pickle.load(f)

# content = []
#
# for id, x in enumerate(data):
#     if len(x.split()) <= 100:
#         content.append(x)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    '/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/llama-7b-armor-new',
    torch_dtype=torch.float16,
).cuda()



generations = []
#generator = pipeline("text-generation", model='/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/llama-7b-armor-new', tokenizer=tokenizer)
for i,sentence in tqdm(enumerate(train_dataset)):
    text = sentence['question']
    input = tokenizer.encode(text, return_tensors='pt').cuda()
    outputs = model.generate(input, do_sample=True, top_p=0.9, top_k=0, max_length=1024, num_return_sequences=1)
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    output = output[0].replace(text, '')
    #outputs = model.generate(sentence, do_sample=True, top_p=0.9, top_k=0, max_length=256, num_return_sequences=1)
    print(i)
    print(output)
    generations.append(output)
# sentence = """Question: I'm designing a RESTful API and I'm trying to work out how I could represent a predicate with OR an operator when querying for a resource.
#
# For example if I had a resource Foo with a property Name, how would you search for all Foo resources with a name matching "Name1" OR "Name2"?
#
# This is straight forward when it's an AND operator as I could do the following:
# <http://www.website.com/Foo?Name=Name1&Age=19>
#
# The other approach I've seen is to post the search in the body.
#
# Answer:"""
# input = tokenizer.encode(sentence, return_tensors='pt').cuda()
# outputs = model.generate(input, do_sample=True, top_p=0.9, top_k=0, max_length=256, num_return_sequences=1)
# output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# output = output[0].replace(sentence, '')
# print(output)


with open('/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/armor_generations.pkl', 'wb') as f:
    pickle.dump(generations, f)



