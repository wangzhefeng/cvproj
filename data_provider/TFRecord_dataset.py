import tensorflow as tf
import os
import matplotlib.pyplot as plt


# root
root_dir = "/Users/zfwang/project/machinelearning/deeplearning"
# project
project_path = os.path.join(root_dir, "deeplearning/src/tensorflow_src")
# model save
models_path = os.path.join(project_path, "save")
# data
cats_and_dogs_dir = os.path.join(root_dir, "datasets/cats_vs_dogs")
data_dir = os.path.join(root_dir, "datasets/cats_vs_dogs/cats_and_dogs_small")
# train data
train_dir = os.path.join(data_dir, "train")
train_cats_dir = os.path.join(train_dir, "cat")
train_dogs_dir = os.path.join(train_dir, "dog")
# tfrecord
tfrecord_file = os.path.join(cats_and_dogs_dir, "train.tfrecord")


# --------------------------------------------------------------------------------------
# 迭代读取每张图片，建立 tf.train.Feature 字典和 tf.train.Example 对象，序列化并写入 TFRecord
# --------------------------------------------------------------------------------------
# 训练数据
train_cat_filenames = [os.path.join(train_cats_dir, filename) for filename in os.listdir(train_cats_dir)]
train_dog_filenames = [os.path.join(train_dogs_dir, filename) for filename in os.listdir(train_dogs_dir)]
train_filenames = train_cat_filenames + train_dog_filenames
train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)

# 保存数据为 TFRecord
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        image = open(filename, "rb").read()
        # 建立 tf.train.Feature 字典
        feature = {
            # 图片是一个 Byte 对象
            "image": tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
            "label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
        }
        # 通过字典建立 Example
        example = tf.train.Example(features = tf.train.Features(feature = feature))
        # 将 Example 序列化并写入 TFRecord 文件
        writer.write(example.SerializeToString())


# --------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------
def _parse_example(example_string):
    """
    将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    """
    # 定义 Feature 结构，告诉解码器每个 Feature 的类型是什么
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    # 解码 JPEG 图片
    feature_dict["image"] = tf.io.decode_jpeg(feature_dict["image"])
    return feature_dict["image"], feature_dict["label"]

# 读取 TFRecord 文件
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = raw_dataset.map(_parse_example)

for image, label in dataset:
    plt.title("cat" if label == 0 else "dog")
    plt.imshow(image.numpy())
    plt.show()
