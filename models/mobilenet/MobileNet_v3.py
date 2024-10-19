# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MobileNet_v3.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032911
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torchvision.models import (
    # mobilenetv3,
    mobilenet_v3_small,
    # mobilenet_v3_large,
    # MobileNetV3,
    MobileNet_V3_Small_Weights,
    # MobileNet_V3_Large_Weights,
    get_model_builder,
)
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


net = mobilenet_v3_small(
    weights = MobileNet_V3_Small_Weights.DEFAULT,
    progress = False,
)
net.eval()




# 测试代码 main 函数
def main():
    # ------------------------------
    # data download and load
    # ------------------------------
    from data_provider.data_download import image_download
    # data
    img_url = "https://cdn.shopify.com/s/files/1/0024/9803/5810/products/583309-Product-0-I-637800179303038345.jpg"
    img_name = "583309-Product-0-I-637800179303038345.jpg"
    img_path = "D:\projects\computer_vision\cvproj\data\cv_clf_imgs\\"
    image, image_tensor = image_download(img_url, img_name, img_path)
    print(image_tensor)
    # ------------------------------
    # data preprocess
    # ------------------------------
    # crop_size = [224]
    # resize_size = [256]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # interpolation = InterpolationMode.BILINEAR
    
    # ------------------------------
    # model predict - MobileNet
    # ------------------------------
    # img preprocess
    mobilenet_preprocess_img = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    img_mobilenet = mobilenet_preprocess_img(image_tensor).unsqueeze(dim = 0)
    print(f"Input Image shape: {img_mobilenet.shape}")
    
    # img predict
    pred_mobilenet = net(img_mobilenet)
    print(f"Output Image shape: {pred_mobilenet.shape}")
    
    # target labels
    cates_mobilenet = MobileNet_V3_Small_Weights.DEFAULT.meta["categories"]
    
    preds_mobilenet = [
        cates_mobilenet[idx]
        for idx in pred_mobilenet.argsort()[0].numpy()
    ][::-1][:3]
    for pred in preds_mobilenet:
        print(pred)
    # ------------------------------
    # prediction visualization
    # ------------------------------
    fig = plt.figure(figsize = (20, 6))
    for i, img in enumerate([image]):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(img)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.text(0, 0, f"{preds_mobilenet[i]}\n")
    plt.show() 

if __name__ == "__main__":
    main()
