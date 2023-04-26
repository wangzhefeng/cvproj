# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils_text.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-25
# * Version     : 0.1.042519
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os

import tensorflow as tf
import tensorflow_datasets as tfds

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class TextData:

    def __init__(self) -> None:
        pass

    def get_text_paths(self, data_url, file_names):
        """
        获取文本数据文件路径地址
        """
        all_text_paths = []
        for file_name in file_names:
            text_dir = tf.keras.utils.get_file(file_name, origin = data_url + file_name)
            all_text_paths.append(text_dir)
        return all_text_paths

    def labeler(self, example, index):
        return example, tf.cast(index, tf.int64)

    def get_labeled_data(self, parent_dir, file_names, buffer_size):
        labeled_data_sets = []
        # parent_dir = os.path.dirname(all_text_paths[0])
        for i, file_name in enumerate(file_names):
            lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
            labeled_dataset = lines_dataset.map(lambda ex: self.labeler(ex, i))
            labeled_data_sets.append(labeled_dataset)
        
        all_labeled_data = labeled_data_sets[0]
        for labeled_dataset in labeled_data_sets[1:]:
            all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
        all_labeled_data = all_labeled_data.shuffle(buffer_size, reshuffle_each_iteration = False)
        return all_labeled_data

    def build_token_set(self, all_labeled_data):
        """
        通过将文本标记为单独的单词集合来构建词汇表
        """
        tokenizer = tfds.deprecated.text.Tokenizer()
        vocabulary_set = set()
        for text_tensor, _ in all_labeled_data:
            some_tokens = tokenizer.tokenize(text_tensor.numpy())
            vocabulary_set.update(some_tokens)
        print(f"vocabulary size is {len(vocabulary_set)}")
        return vocabulary_set

    def get_encoded_data(self, vocabulary_set, text_tensor, label):
        encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set)
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(self, text, label):
        encoded_text, label = tf.py_function(
            self.get_encoded_data, 
            inp = [text, label], 
            Tout = (tf.int64, tf.int64)
        )
        encoded_text.set_shape([None])
        label.set_shape([])
        return encoded_text, label

    def split_train_test(self, all_encoded_data, take_size, buffer_size, batch_size):
        train_data = all_encoded_data \
            .skip(take_size) \
            .shuffle(buffer_size) \
            .padded_batch(batch_size)
        test_data = all_encoded_data \
            .take(take_size) \
            .padded_batch(batch_size)
        return train_data, test_data




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
