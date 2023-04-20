# -*- coding: utf-8 -*-


# ***************************************************
# * File        : pre_train_word_embedding.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-27
# * Version     : 0.1.032719
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import math
import random
import matplotlib.pyplot as plt

import torch
from torch import nn
from d2l import torch as d2l


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]



"""
# 核心概念：
    text
    sentence
    line
    token
    vocab
"""

# ------------------------------
# data
# ------------------------------
def read_ptb():
    """
    将 PTB 数据集记载到文本行
    """
    # data download
    d2l.DATA_HUB["ptb"] = (d2l.DATA_URL + "ptb.zip", "319d85e578af0cdc590547f26231e4e31cdf1e42")
    # data extract
    data_dir = d2l.download_extract("ptb")
    # train dataset
    with open(os.path.join(data_dir, "ptb.train.txt")) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split("\n")]

sentences = read_ptb()
print(f"Number of sentences: {len(sentences)}")
print(f"sentences[0]: {sentences[0]}")

# ------------------------------
# vocab
# ------------------------------
vocab = d2l.Vocab(sentences, min_freq = 10)
print(f"Vocab size: {len(vocab)}")

# ------------------------------
# subsample(下采样)
# ------------------------------
def subsample(sentences, vocab):
    """
    下采样高频词
    """
    # 排除未知词元
    sentences = [
        [token for token in line if vocab[token] != vocab.unk] 
        for line in sentences
    ]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())
    # 如果在下采样期间保留词元，则返回 True
    def keep(token):
        return (random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens))

    subsampled = ([
        [token for token in line if keep(token)] 
        for line in sentences
    ])
    return subsampled, counter

subsampled, counter = subsample(sentences, vocab)
print(f"Number of subsampled sentences: {len(subsampled)}")

# 下采样前后每句话的词元数量的直方图
# d2l.show_list_len_pair_hist(
#     ["origin", "subsampled"], 
#     "num tokens per sentence", "count", 
#     sentences, subsampled
# )
# plt.show()

# ------------------------------
# Pre-train word2vec
# ------------------------------
batch_size = 512
max_windows_size = 5
num_noise_words = 5
data_iter, vocab = d2l.load_data_ptb(
    batch_size,
    max_windows_size,
    num_noise_words,
)


class SkipGram():

    def __init__(self, num_embeddings = 20, embedding_dim = 4) -> None:
        # embedding layer
        self.embed = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = embedding_dim)
        print(f"Parameter embedding_weight: ({self.embed.weight.shape}, dtype={self.embed.weight.dtype})")

    def skip_gram(self, center, contexts_and_negatives, embed_v, embed_u):
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        
        return pred





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
