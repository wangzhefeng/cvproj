# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import tensorflow_datasets as tfds


def get_text_paths(data_url, file_name):
    """
    获取文本数据文件路径地址
    """
    all_text_paths = []
    for name in file_name:
        text_dir = tf.keras.utils.get_file(name, origin = data_url + name)
        all_text_paths.append(text_dir)

    return all_text_paths


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


def get_labeled_data(parent_dir, file_names, buffer_size):
    """
    
    """
    labeled_data_sets = []
    # parent_dir = os.path.dirname(all_text_paths[0])
    for i, file_name in enumerate(file_names):
        lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)
    
    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
    all_labeled_data = all_labeled_data.shuffle(buffer_size, reshuffle_each_iteration = False)
    
    return all_labeled_data


def build_token_set(all_labeled_data):
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


def get_encoded_data(text_tensor, label):
    # encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set)
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(text, label, vocabulary_set):
    encoded_text, label = tf.py_function(get_encoded_data, inp = [text, label], Tout = (tf.int64, tf.int64))
    encoded_text.set_shape([None])
    label.set_shape([])
    
    return encoded_text, label


def split_train_test(all_encoded_data, take_size, buffer_size, batch_size):
    train_data = all_encoded_data.skip(take_size).shuffle(buffer_size).padded_batch(batch_size)
    test_data = all_encoded_data.take(take_size).padded_batch(batch_size)
    
    return train_data, test_data
