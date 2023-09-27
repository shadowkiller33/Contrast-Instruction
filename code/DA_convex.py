from datasets import load_dataset
import json
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from tqdm import tqdm
j_list = []
k_list = []
tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_cq2/share_1603164/data/huggingface_models/princeton-nlp/sup-simcse-roberta-base")
model = AutoModel.from_pretrained("/apdcephfs_cq2/share_1603164/data/huggingface_models/princeton-nlp/sup-simcse-roberta-base")
dataset = load_dataset('json', data_files='/apdcephfs_cq2/share_1603164/data/lingfengshen/data/wmt/data.json', split='train')
train_data = dataset.train_test_split(test_size=0.2)['train'].select(range(500))
for x in train_data['response_j']:
    j_list.append(x)
for x in train_data['response_k']:
    k_list.append(x)

data = []
number = 0
for _,i in tqdm(enumerate(range(len(j_list)))):
    j_text = j_list[i]
    k_text = k_list[i]

    #aug = naw.SpellingAug()
    case = {}
    length = len(j_text.split())
    aug = naw.SpellingAug(aug_max=int(0.05*length))
    augmented_texts = aug.augment(j_text, n=5)

    inputs = tokenizer(augmented_texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().detach().numpy()


    # X = rng.random((20, 10))
    pca = PCA(n_components=2,whiten=False)
    pca.fit(embeddings)
    pca_embedding = pca.transform(embeddings)
    #print(pca_embedding.shape)
    hull = ConvexHull(pca_embedding)
    vertices = hull.vertices
    number += len(vertices)
    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    for vertex in vertices:
        case['response_j'] = augmented_texts[vertex]
        case['response_k'] = k_text
        data.append(case)
        #print(augmented_texts[vertex])

print(number/len(j_list))
with open('/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/DA_wmt.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)