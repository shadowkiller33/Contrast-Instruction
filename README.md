# Contrast Instructions

This is the official code repository for the paper [The Trickle-down Impact of Reward (In-)consistency on RLHF](https://arxiv.org/pdf/2309).

## Contrast Instructions Benchmark
The four contrast instructions benchmark datasets can be found under [`data/contrast_instructions`](https://github.com/shadowkiller33/Contrast-Instruction/tree/master/data/contrast_instructions).
| Dataset | Task | Num. Examples | Link to file |
| --- | --- | --- | ---|
|`StackExchange` | Question Answering | 1188 | [`stack_contrast.json`](https://github.com/shadowkiller33/Contrast-Instruction/blob/master/data/contrast_instructions/stack_contrast.json)|
|`WMT`| Machine Translation | 612 | [`wmt_contrast.json`](https://github.com/shadowkiller33/Contrast-Instruction/blob/master/data/contrast_instructions/wmt_contrast.json) |
| `Twitter` | Paraphrase Identification | 289 | [`para_contrast.json`](https://github.com/shadowkiller33/Contrast-Instruction/blob/master/data/contrast_instructions/para_contrast.json) |
| `RealSumm` | Summarization | 36 | [`sum_contrast.json`](https://github.com/shadowkiller33/Contrast-Instruction/blob/master/data/contrast_instructions/sum_contrast.json) |

Each example in the json file looks like this (example from `WMT`) -- 
```
{
  "query": "这一切，身在海外的华人华侨感受更为深刻。",
  "retrieval": "身在上海，是一种亲历才懂的情感。",
  "query_response_k": "All this, the overseas Chinese living overseas feel more deeply.",
  "query_response_j": "All of this, the overseas Chinese feel even more deeply.",
  "retrieval_response_k": "Being in Shanghai is a kind of emotion that you know.",
  "retrieval_response_j": "Being in Shanghai is a kind of emotion that can only be understood through experience."
}
```
`query` and `retrieval` correspond to the two (inputs of) the instructions. `*_response_k` is the human preferred response for `query` and `retrieval` respectively. `*_response_j` is a less preferred response, that is NOT used in our reward consistency metrics. 

## Human preference data
We release the human preference data + splits for `WMT`, `Twitter` and `RealSumm` under [`data/human_preferences`](https://github.com/shadowkiller33/Contrast-Instruction/tree/master/data/human_preferences). The `StackExchange` dataset can be found on Hugging Face datasets -- [`HuggingFaceH4/stack-exchange-preferences`](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences). 


## Running Evaluation
WIP; We are still cleaning + organizing code for release. Please reach out to `lshen30[at]jhu.edu` and `sihaoc[at]cis.upenn.edu` for questions. 



