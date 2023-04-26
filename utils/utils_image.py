# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils_image.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-04-05
# * Version     : 0.1.040523
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

import sys
import pathlib
import random
from pathlib import Path
from typing import List

import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import torch
from IPython.core.interactiveshell import InteractiveShell

print(f"TensorFlow version: {tf.__version__}")
InteractiveShell.ast_node_interactivity = "all"
AUTOTUNE = tf.data.experimental.AUTOTUNE

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# %matplolib inline
# %config InlineBackend.figure_format = "svg"


def resize_and_pad_image(image, width, height):
    img = image
    w, h = img.size 
    ratio = w / float(h)
    imgH, imgW = height, width
    
    if imgH * ratio >= imgW:
        resized_w = imgW
        resized_h = np.floor(imgW / ratio).astype('int32')
        resized_image = img.resize((resized_w, resized_h))
        resized_arr = np.array(resized_image).astype('float32')

        if img.mode == 'L':
            padding_arr = np.zeros((imgH, imgW), dtype = np.float32)
        else:
            assert img.mode == 'RGB'
            padding_arr = np.zeros((imgH, imgW,3), dtype = np.float32)
         
        padding_arr[:resized_h, :] = resized_arr
        padding_im = PIL.Image.fromarray(padding_arr.astype(np.uint8))
        box_factor = resized_h / h
        return  padding_im
    else:
        resized_h = imgH
        resized_w = np.floor(imgH * ratio).astype('int32')
        resized_image = img.resize((resized_w, resized_h))
        resized_arr = np.array(resized_image).astype('float32')
        
        if img.mode == 'L':
            padding_arr = np.zeros((imgH, imgW), dtype = np.float32)
        else:
            assert img.mode == 'RGB'
            padding_arr = np.zeros((imgH, imgW, 3), dtype = np.float32)
         
        padding_arr[:, :resized_w] = resized_arr
        padding_im = PIL.Image.fromarray(padding_arr.astype(np.uint8))
        box_factor = resized_w / w
        return padding_im


def plot_ds_train(ds_train):
    plt.figure(figsize = (8, 8))
    for i in range(9):
        img, label = ds_train[i]
        img = img.permute(1, 2, 0)
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img.numpy())
        ax.set_title("label = %d" % label.item())
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_tf_ds_train(ds_train):
    plt.figure(figsize = (8, 8))
    for i, (img, label) in enumerate(ds_train.unbatch().take(9)):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img.numpy())
        ax.set_title("label = %d" % label)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def show_images(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    """
    Plot a list of images.
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def set_matplotlib_font(font_size = 15):
    import shutil

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    shutil.rmtree(mpl.get_cachedir())
    simhei_file = Path(__file__).parent/"assets/SimHei.ttf"
    ttf_dir = Path(mpl.__path__[0])/'mpl-data'/'fonts'/'ttf'
    shutil.copy(str(simhei_file),str(ttf_dir))
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.size'] = str(font_size)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"set matplotlib font to SimHei and fontsize to {font_size}")


def rgb_img_view(dataset):
    plt.figure(figsize = (8, 8))
    for i in range(9):
        # data
        img, label = dataset[i]
        img = torch.squeeze(img).numpy()
        # plot
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"label = {label}")
    plt.show()


class _Colors:

    def __init__(self):
        self.hexs = (
            'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', 
            '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', 
            '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7'
        )
        self.palette = [self.hex2rgb(f'#{c}') for c in self.hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr = False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c
    
    def get_hex_color(self, i):
        return self.hexs[int(i)%self.n]

    @staticmethod
    def hex2rgb(h):
        """
        # rgb order (PIL)

        Args:
            h (_type_): _description_

        Returns:
            _type_: _description_
        """
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class Annotator:

    def __init__(self, img, line_width = None, font_size = None):
        assert np.asarray(img).data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to input images.'
        self.img = img if isinstance(img, PIL.Image.Image) else PIL.Image.fromarray(img)
        
        try:
            path = Path(__file__)
            font = str(path.parent/"assets/Arial.ttf")
            size = font_size or max(round(sum(self.img.size) / 2 * 0.035), 12)
            self.font = PIL.ImageFont.truetype(font, size)
        except Exception as err:
            print(err, file = sys.stderr)
            self.font = PIL.ImageFont.load_default()
         
        self.lw = line_width or max(round(sum(self.img.size) / 2 * 0.003), 2)  
        self.pil_9_2_0_check = True if PIL.__version__ >= '9.2.0' else False

    def add_box_label(self, box, label = '', color = (128, 128, 128), txt_color = (255, 255, 255)):
        # Add one xyxy box to image with label
        self.draw = PIL.ImageDraw.Draw(self.img)
        self.draw.rectangle(box, width = self.lw, outline = color)  # box
        if label:
            if self.pil_9_2_0_check:
                _, _, w, h = self.font.getbbox(label)  # text width, height (New)
            else:
                w, h = self.font.getsize(label)  # text width, height (Old, deprecated in 9.2.0)
            outside = box[1] - h >= 0  # label fits outside box
            self.draw.rectangle(
                (
                    box[0], 
                    box[1] - h if outside else box[1], 
                    box[0] + w + 1,
                    box[1] + 1 if outside else box[1] + h + 1
                ),
                fill = color,
            )
            # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
            self.draw.text(
                (
                    box[0], 
                    box[1] - h if outside else box[1]
                ), 
                label, 
                fill = txt_color, 
                font = self.font
            )

    def add_mask(self, mask, color, alpha=0.5):
        # Add one mask to image
        img = np.asarray(self.img).copy()

        im1_shape = mask.shape
        im0_shape = img.shape

        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2 

        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
        mask = mask[top:bottom, left:right]

        mask_img = PIL.Image.fromarray((255 * mask).astype(np.uint8), mode = 'L')
        mask = np.array(mask_img.resize(self.img.size, resample = PIL.Image.BILINEAR)) >= 1
        img[mask] = img[mask] * (1-alpha) + np.array(color) * alpha
        self.img = PIL.Image.fromarray(img)


def plot_detection(img, boxes, class_names, min_score = 0.2):
    """
    图像检测

    Args:
        img (_type_): _description_
        boxes (_type_): _description_
        class_names (_type_): _description_
        min_score (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    colors = _Colors()
    annotator = Annotator(img)
    for *box, conf, cls in reversed(boxes):
        if conf > min_score:
            label = f'{class_names[int(cls)]} {conf:.2f}'
            annotator.add_box_label(box, label, color = colors(cls))
     
    return annotator.img 


def plot_instance_segmentation(img, boxes, masks, class_names, min_score = 0.2):
    """
    实例分割

    Args:
        img (_type_): _description_
        boxes (_type_): _description_
        masks (_type_): _description_
        class_names (_type_): _description_
        min_score (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    colors = _Colors()
    annotator = Annotator(img)
    for (*box, conf, cls), mask in zip(reversed(boxes), reversed(masks)):
        if conf > min_score:
            label = f'{class_names[int(cls)]} {conf:.2f}'
            annotator.add_box_label(box, label, color = colors(cls))
            annotator.add_mask(mask.cpu().numpy(), colors(cls))

    return annotator.img


def vis_detection(image,
                  prediction,
                  class_names = None,
                  min_score = 0.2,
                  figsize = (16, 16),
                  linewidth = 2,
                  color = 'lawngreen'):
    # TODO
    boxes = np.array(prediction['boxes'].tolist()) 
    scores = np.array(prediction['scores'].tolist()) if 'scores' in prediction else [1.0 for _ in boxes]
    labels = np.array(prediction['labels'].tolist()) if 'labels' in prediction else None
    classes = [class_names[x] for x in labels] if (
        class_names is not None and labels is not None
    ) else ['object'] * len(boxes)
    # 
    import matplotlib.pyplot as plt
    image = np.array(image, dtype = np.uint8)
    fig = plt.figure(figsize = figsize)
    plt.axis("off")
    plt.imshow(image);
    ax = plt.gca()
    for i, (box, name, score) in enumerate(zip(boxes, classes, scores)):
        if score >= min_score:
            text = "{}: {:.2f}".format(name, score)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle(
                [x1, y1], w, h, 
                fill = False, 
                edgecolor = color, 
                linewidth = linewidth
            )
            ax.add_patch(patch)
            ax.text(
                x1, y1, text,
                bbox = {"facecolor": color, "alpha": 0.8},
                clip_box = ax.clipbox,
                clip_on = True,
            )
    plt.show();


def joint_imgs_row(img1, img2):
    size1 = img1.size
    size2 = img2.size
    joint = PIL.Image.new('RGB', (size1[0] + size2[0], max(size1[1], size2[1])))
    loc1, loc2 = (0, 0), (size1[0], 0)
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)

    return joint 


def joint_imgs_col(img1, img2):
    size1 = img1.size
    size2 = img2.size
    joint = PIL.Image.new('RGB', (max(size1[0], size2[0]), size1[1] + size2[1]))
    loc1, loc2 = (0, 0), (0, size1[1])
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)

    return joint 


def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    return color_map

def gray2pseudo(gray_img):
    color_map = get_color_map_list(256)
    gray_p = gray_img.convert('P')
    gray_p.putpalette(color_map)

    return gray_p 


def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)

    return img


def text2img(text):
    path = Path(__file__)
    simhei = path.parent/"assets/SimHei.ttf"
    lines  = len(text.split("\n")) 
    image = PIL.Image.new("RGB", (800, lines * 20), (255, 255, 255))
    draw = PIL.ImageDraw.Draw(image)
    font = PIL.ImageFont.truetype(str(simhei), 18)
    draw.text((0, 0), text, font = font, fill = "#000000")

    return image


def img2tensor(image):
    from torchvision.transforms import ToTensor
    tensor = ToTensor()(np.array(image))

    return tensor


def image_download(data_url: str, data_fname: str, untar: bool = False) -> str:
    """
    下载图片数据
    
    Args:
        data_url (str): _description_ 
            例如: 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
        data_fname (str): _description_
            例如: flower_photos
        untar (bool, optional): _description_. Defaults to False.

    Returns:
        str: 数据文件所在路径
    """
    data_root_origin = tf.keras.utils.get_file(
        origin = data_url,
        fname = data_fname,
        untar = untar
    )
    data_root = pathlib.Path(data_root_origin)
    for item in data_root.iterdir():
        # print(item)
        pass
    
    return data_root


def image_paths(data_root: str, is_shuffle = False) -> List:
    """
    获取所有图像地址
    """
    all_image_paths = list(data_root.glob("*/*"))
    all_image_paths = [str(path) for path in all_image_paths]
    if is_shuffle:
        random.shuffle(all_image_paths)
    # 图像数量
    image_count = len(all_image_paths)
    print(f"image_count={image_count}")

    return all_image_paths


def image_load_preprocessing(img_path):
    """
    加载和格式化图像数据
    Args:
        img_path:
        size:
    """
    img_raw = tf.io.read_file(img_path)
    # img_tensor = tf.image.decode_image(img_raw, channels = 3)
    img_tensor = tf.image.decode_jpeg(img_raw, channels = 3)
    # img_tensor = tf.image.decode_and_crop_jpeg()
    # img_tensor = tf.image.decode_bmp()
    # img_tensor = tf.image.decode_gif()
    # img_tensor = tf.image.decode_png()
    img_resized = tf.image.resize(img_tensor, [192, 192])
    image = img_resized / 255.0

    return image


def image_preprocessing(img_path):
    """
    加载和格式化图像数据
    Args:
        img_path:
    """
    img_raw = tf.io.read_file(img_path)
    # img_tensor = tf.image.decode_image(img_raw, channels = 3)
    img_tensor = tf.image.decode_jpeg(img_raw, channels = 3)
    # img_tensor = tf.image.decode_and_crop_jpeg()
    # img_tensor = tf.image.decode_bmp()
    # img_tensor = tf.image.decode_gif()
    # img_tensor = tf.image.decode_png()
    img_resized = tf.image.resize(img_tensor, [192, 192])
    image = img_resized / 255.0

    return image


def image_display(img_path):
    """
    展示图片
    """
    display.display(display.Image(img_path))


def image_labels(data_root, all_image_paths):
    all_label_names = sorted(item.name for item in data_root.glob("*/") if item.is_dir())
    all_label_dict =  dict((name, index) for index, name in enumerate(all_label_names))
    all_image_labels = [all_label_dict[pathlib.Path(path).parent.name] for path in all_image_paths]

    return all_label_names, all_label_dict, all_image_labels


def image_display_v2(img_path, img_label, all_label_names):
    image = image_load_preprocessing(img_path)
    plt.imshow(image)
    plt.grid(False)
    # plt.xlabel()
    plt.title(all_label_names[img_label].title())



# data_augmentation = tf.keras.models.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal", input_shape = (img_height, img_width, 3)),
#     layers.experimental.preprocessing.RandomRotation(0.1),
#     layers.experimental.preprocessing.RandomZoom(0.1),
# ])




# 测试代码 main 函数
def main():
    data_root = image_download(
        data_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        data_fname = "flower_photos",
        untar = True
    )
    print(data_root)
    all_image_paths = image_paths(data_root)
    print(all_image_paths[:5])

    def caption_image(image_path):
        attributions = (data_root/"LICENSE.txt").open(encoding = "utf-8").readlines()[4:]
        attributions = [line.split(" CC-BY") for line in attributions]
        attributions = dict(attributions)
        image_rel = pathlib.Path(image_path).relative_to(data_root)
        return "Image (CC BY 2.0) " + " - ".join(attributions[str(image_rel)].split(" - ")[:-1])

    for n in range(3):
        image_path = random.choice(all_image_paths)
        image_display(img_path = image_path)
        print(caption_image(image_path))


if __name__ == "__main__":
    main()
