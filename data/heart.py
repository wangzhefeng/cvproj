# -*- coding: utf-8 -*-

# ***************************************************
# * File        : heart.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042423
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# TODO
class DataLoader():

    def __init__(self):
        path = tf.keras.utils.get_file(
            "nietzsche.txt", 
            origin = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
        )
        with open(path, encoding = "utf-8") as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index + seq_length])
            next_char.append(self.text[index + seq_length])
        return np.array(seq), np.array(next_char) # [batch_size, seq_length], [num_batch]


def heart():
    # read dataset
    csv_file = tf.keras.utils.get_file(
        "heart.csv", 
        origin = "https://storage.googleapis.com/applied-dl/heart.csv"
    )
    df = pd.read_csv(csv_file)
    # data preprocessing
    df["thal"] = pd.Categorical(df["thal"])
    df["thal"] = df.thal.cat.codes
    # target
    target = df.pop("target")

    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    return dataset




# 测试代码 main 函数
def main():
    dataset = heart()
    for features, target in dataset.take(5):
        pass

if __name__ == "__main__":
    main()

