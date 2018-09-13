import math
import random

import cv2
import matplotlib.pyplot as plt
import keras
import os

from PIL import Image

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

import h5py


def read_pic_path(pic_src):
    """
    获取分类类别
    :param pic_src: 训练图像文件路径
    :return:
    """
    # 得出子目录
    contents = os.listdir(pic_src)
    # 有子文件的目录为有效的
    classes = [each for each in contents if os.path.isdir(pic_src + each)]

    return classes


def get_all_files_path(pic_src):
    """
    获取图像路径、图像标签、图像类别
    :param pic_src: 训练图像文件路径
    :return:
    """
    all_images = []
    all_labels = []

    classes = read_pic_path(pic_src)
    for index, class_name in enumerate(classes):
        class_path = pic_src + class_name
        for img_name in os.listdir(pic_src + class_name):
            img_path = class_path + '/' + img_name
            all_images.append(img_path)
            all_labels.append(index)

    return all_images, all_labels, classes


def read_img_2_folder(imgs_list, img_size, dst_folder):
    """
    将图像 Resize，并复制到指定文件夹
    :param imgs_list: 原始文件路径
    :param img_size: resize 后的图像尺寸
    :param dst_folder: 目标文件路径
    :return:
    """
    
    # 使用padding方式填充原图为正方形，并resize到目标大小
    def resize_img(img, size):
        longer_side = max(img.size)
        horizontal_padding = (longer_side - img.size[0]) / 2
        vertical_padding = (longer_side - img.size[1]) / 2

        img_croped = img.crop(
            (
                -((299 - img.size[0]) / 2),
                -((299 - img.size[1]) / 2),
                img.size[0] + (299 - img.size[0]) / 2,
                img.size[1] + (299 - img.size[1]) / 2
            )
        )
        img_resized = img_croped.resize((size, size))
        return img_resized

    for index, img_path in enumerate(imgs_list):
        img = Image.open(img_path)
        # 有些图片是png（RGBA），会导致后面出错，在这里统一转换成RGB
        if img.mode != 'RGB':
            img = img.convert("RGB")
        img_resized = image_resize(img, img_size)

        path_elements = img_path.split('/')
        new_sub_path = path_elements[-2] + '/' + path_elements[-1]
        new_sub_folder = path_elements[-2]
        new_path = dst_folder + new_sub_path
        new_folder = dst_folder + new_sub_folder

        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        img_resized.save(new_path, 'png')
        img.close()


def data_preprocess(pic_src, train_ratio, img_size, data_dst):
    """
    图像预处理
    :param pic_src: 训练图像原始路径
    :param train_ratio: 训练图像占比
    :param img_size: resize后的图像尺寸
    :param data_dst: 训练图像目标路径
    :return:
    """
    # 读取所有路径，并乱序排列
    all_images, all_labels, classes = get_all_files_path(pic_src)
    all_images, all_labels = shuffle_data_list(all_images, all_labels)

    # 获取切割num
    data_num = len(all_labels)
    train_num = int(data_num * train_ratio)

    # 切割label
    train_labels = all_labels[:train_num]  # 切分出 从 0 到第train_num-1个（或者说前train_num个）形成新数组
    valid_labels = all_labels[train_num:]  # 先切分出从第train_num个到最后一个

    # 切割image
    train_images = all_images[:train_num]
    valid_images = all_images[train_num:]

    # 存储到h5py中
    read_img_2_folder(train_images, img_size, data_dst[0])
    read_img_2_folder(valid_images, img_size, data_dst[1])


def shuffle_data_list(data_list, label_list):
    """
    对图像进行 Shuffle 处理
    :param data_list: 图像文件名列表
    :param label_list: 图像标签列表，与data_list相对应
    :return:
    """
    index = [i for i in range(len(data_list))]
    random.shuffle(index)
    data_list = [data_list[each] for each in index]
    label_list = [label_list[each] for each in index]
    return data_list, label_list


def image_resize(batch_image_lst, output, size=299):
    """
    使用Padding方式对图像进行填充，统一 resize 为 299*299*3
    :param batch_image_lst: 待 resize 处理图像 Image path list
    :param output: resized后的文件输出路径
    :param size: resize后的图像尺寸
    :return:
    """
    images = np.zeros((len(batch_image_lst), size, size, 3), dtype=np.uint8)
    for i in range(len(batch_image_lst)):
        path = batch_image_lst[i]
        name = os.path.basename(path)
        label = os.path.dirname(path).split("/")[-1]

        img = cv2.imread(path)
        img = img[:, :, ::-1]
        img = cv2.resize(img, (299, 299))

        resized_image_path = os.path.join(output, label)
        if not os.path.exists(resized_image_path):
            os.makedirs(resized_image_path)

        # 保存resized图像，供后期使用
        resized_image_path = os.path.join(resized_image_path, name)
        cv2.imwrite(resized_image_path, img)

        images[i] = img
    return images


class LossHistory(keras.callbacks.Callback):
    """
    处理模型训练输出的log，并生成 loss 与 accuracy 曲线
    """
    def __init__(self, model):
        self.model = model
        self.epoch = 0

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
        self.epoch = 0
        
    def on_epoch_begin(self, batch, logs={}):
        self.epoch += 1

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        # self.model.save('xception_finetune_epoch_%s.h5' % self.epoch)
        

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xticks(iters, iters)
        plt.legend()
        plt.show()
        
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.plot(iters, self.accuracy[loss_type], 'g', label='train accuracy')
        plt.plot(iters, self.val_acc[loss_type], 'k', label='val accuracy')
        plt.grid(True)
        plt.xticks(iters, iters)
        plt.legend()
        plt.show()


def image_list_display(img_list):
    """
    显示图像列表
    :param img_list: 待显示图像路径list
    :return:
    """
    fig = plt.figure(figsize=(15, 3 * math.ceil(len(img_list) / 5)))
    for i in range(0, len(img_list)):
        img = cv2.imread(img_list[i])
        img = img[:, :, ::-1]  # BGR->RGB
        ax = fig.add_subplot(math.ceil(len(img_list) / 5), 5, i + 1)
        ax.set_title(os.path.basename(img_list[i]))
        ax.set_xticks([])
        ax.set_yticks([])
        img = cv2.resize(img, (128, 128))
        ax.imshow(img)
    plt.show()


def generate_feature_vector(MODEL, image_size, train_data_path, valid_data_path, lambda_func=None):
    """
    提取图像特征
    :param MODEL: 模型
    :param image_size: 图像大小，eg (224, 224, 3)
    :param train_data_path: 训练数据路径
    :param valid_data_path: 验证数据路径
    :param lambda_func: 回调函数
    :return:
    """
    with tf.device('/cpu:0'):
        input_tensor = Input(image_size)
        x = input_tensor
        if lambda_func:
            x = Lambda(lambda_func)(x)

        base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
        model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

        # 图像增强处理
        gen = ImageDataGenerator()
        train_generator = gen.flow_from_directory(train_data_path, image_size, shuffle=False, batch_size=16)
        valid_generator = gen.flow_from_directory(valid_data_path, image_size, shuffle=False, batch_size=16,
                                                  class_mode=None)

        train = model.predict_generator(train_generator, train_generator.nb_sample)
        valid = model.predict_generator(valid_generator, valid_generator.nb_sample)

        # 将特征写入文件
        with h5py.File("feature_%s.h5" % MODEL.func_name) as h:
            h.create_dataset("train", data=train)
            h.create_dataset("valid", data=valid)
            h.create_dataset("label", data=train_generator.classes)


# 将训练图像按照batch_size=64进行分组
def batch_names_list(images_path, batch_size=64):
    """
    对列表按照batch_size进行分组
    :param images_path: 待分组文件列表
    :param batch_size: 分组大小
    """
    names = []
    lst = os.listdir(images_path)
    for begin in range(0, len(lst), batch_size):
        end = min(begin + batch_size, len(lst))
        names.append(lst[begin:end])

    return names


# 批量读取图像
def read_batch_images(path, batch):
    """
    批量读取图像
    :param path: 图像根路径
    :param batch: 文件名列表
    """
    lst = []
    for item in batch:
        lst.append(Image.open(os.path.join(path, item)))

    return lst


# 猫狗预测器
def pred_pet(model, img_path_list, top_num, preprocess_input, decode_predictions, batch_size=32):
    ret = []
    for batch_imgpath_list in batch_img(img_path_list, batch_size):
        X = read_batch_img(batch_imgpath_list)
        X = preprocess_input(X)
        preds = model.predict(X)
        dps = decode_predictions(preds, top=top_num)
        for index in range(len(dps)):
            for i, val in enumerate(dps[index]):
                if (val[0] in Dogs) and ('dog' in batch_imgpath_list[index]):
                    ret.append(True)
                    break
                elif (val[0] in Cats) and ('cat' in batch_imgpath_list[index]):
                    ret.append(True)
                    break
                if i == len(dps[index]) - 1:
                    ret.append(False)
    return ret


class FilesScanner(object):
    """
    获取文件列表工具类
    """

    def __init__(self, files_path, suffix=None):
        """

        :param files_path: 待扫描文件路径
        :param suffix: 所需文件后缀，默认为空，即获取该路径下所有文件
        """
        self.files_path = files_path

        files = []
        if os.path.isfile(files_path):
            if suffix:
                if files_path.endswith(suffix):
                    files.append(files_path)
            else:
                files.append(files_path)

        if os.path.isdir(files_path):
            for root, dirs, filenames in os.walk(files_path):
                for filename in filenames:
                    if suffix:
                        if filename.endswith(suffix):
                            files.append(os.path.join(root, filename))
                    else:
                        files.append(os.path.join(root, filename))
        # 替换为绝对路径
        files = [os.path.abspath(item) for item in files]

        self.files = files

    def get_files(self):
        return self.files
    
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min'):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


if __name__ == '__main__':
    pass
