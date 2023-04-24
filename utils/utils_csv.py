# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils_csv.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042423
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import functools
import tensorflow as tf

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_csv_path(train_data_url, train_file_name, test_data_url, test_file_name):
    """
    获取 CSV 文件的文件路径
    """
    train_file_path = tf.keras.utils.get_file(f"{train_file_name}", train_data_url)
    test_file_path = tf.keras.utils.get_file(f"{test_file_name}", test_data_url)

    return train_file_path, test_file_path


def get_csv_dataset(file_path, csv_column, label_column, batch_size, num_epochs):
    """
    读取 file_path 中的 CSV 文件构建为 tf.data.Dataset
    """
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size = batch_size,
        label_name = label_column,
        na_value = "?",
        num_epochs = num_epochs,
        column_names = csv_column,
        # select_columns = csv_column,
        ignore_errors = True,
    )
    
    return dataset


def get_categorical_data(categories):
    """
    构建一个 tf.feature_column.indicator_column 集合，
    每个 tf.feature_column.indicator_column 对应一个分类的列
    Args categories examples:
        CATEGORIES = {
            "sex": ["male", "female"],
            "class": ["First", "Second", "Third"],
            "deck": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "embark_town": ["Cherbourg", "Southhampton", "Queenstown"],
            "alone": ["y", "n"]
        }
    """
    categorical_columns = []
    for feature, vocab in categories.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key = feature, vocabulary_list = vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    return categorical_columns


def get_normalized_data(mean, data):
    """
    标准化连续特征数据
    """
    data = tf.cast(data, tf.float32) * 1 / (2 * mean)
    
    return tf.reshape(data, [-1, 1])


def get_continuous_data(mean):
    """
    使用 normalizer_fn 参数。在传参的时候使用 functools.partial，
    functools.partial 由使用每个列的均值进行标准化的函数构成
    Args mean examples:
        MEANS = {
            'age' : 29.631308,
            'n_siblings_spouses' : 0.545455,
            'parch' : 0.379585,
            'fare' : 34.385399
        }
    """
    numerical_columns = []
    for feature in mean.keys():
        num_col = tf.feature_column.numeric_column(
            feature, 
            normalizer_fn = functools.partial(get_normalized_data, mean[feature])
        )
        numerical_columns.append(num_col)

    return numerical_columns




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

