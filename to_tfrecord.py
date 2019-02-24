# coding=utf-8
import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np

# the number of test
_NUM_TEST = 1000
# random seed
_RANDOM_SEED = 0
# the number of shards
_NUM_SHARDS = 2

DATASET_DIR = './captcha/images/'
TFRECORD_DIR = './captcha/'
# the label filename
LABELS_FILENAME = 'labels.txt'


# define the path and name of the tfrecord file
def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'images_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


# to ensure tfrecord exists or not
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        for shard_id in range(_NUM_SHARDS):
            # define the path and name of the tfrecord file
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

            if not tf.gfile.Exists(output_filename):
                return False
    return True


def _get_filenames_and_classes(dataset_dir):

    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)

        photo_filenames.append(path)

    return photo_filenames


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, label0, label1, label2, label3):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label0': int64_feature(label0),
        'label1': int64_feature(label1),
        'label2': int64_feature(label2),
        'label3': int64_feature(label3),
    }))


def write_label_file(labels_to_class_names, dataset_dir, filenames=LABELS_FILENAME):
    labels_filenames = os.path.join(dataset_dir, filenames)
    with tf.gfile.Open(labels_filenames, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


# 把数据转换为TFRecord格式
def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'test']
    num_per_shard = int(len(filenames) / _NUM_SHARDS)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                # 定义tfrecord文件的路径和名字
                output_filename = _get_dataset_filename(TFRECORD_DIR, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    # 每一个数据块开始的位置
                    start_ndx = shard_id * num_per_shard
                    # 每一个数据块最后的位置
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))

                    for i in range(start_ndx, end_ndx):
                        try:
                            sys.stdout.write("\r>> Conver images %d/%d shard %d" % (i + 1, len(filenames), shard_id))
                            sys.stdout.flush()
                            # 读取图片
                            # image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                            image_data = Image.open(filenames[i])
                            image_data = image_data.resize((224, 224))
                            # 灰度化
                            image_data = np.array(image_data.convert('L'))
                            # 将图片转化为bytes
                            image_data = image_data.tostring()

                            # 获取label
                            labels = filenames[i].split('/')[-1][0:4]
                            num_labels = []
                            for j in range(4):
                                num_labels.append(int(labels[j]))

                            # 生成tfrecord文件
                            example = image_to_tfexample(image_data, num_labels[0], num_labels[1], num_labels[2],
                                                         num_labels[3])
                            tfrecord_writer.write(example.SerializeToString())
                        except IOError as e:
                            print "Could not read: " + filenames[i]
                            print "Error: " + e
                            print "Skip it\n"

    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    if _dataset_exists(DATASET_DIR):
        print "tfrecords文件已经存在"
    else:
        photo_filenames = _get_filenames_and_classes(DATASET_DIR)

        # 把数据切分为训练集和测试集
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[_NUM_TEST:]
        test_filenames = photo_filenames[:_NUM_TEST]

        # 数据转换
        _convert_dataset('train', training_filenames, DATASET_DIR)
        _convert_dataset('test', test_filenames, DATASET_DIR)

        print "生成tfrecord文件"