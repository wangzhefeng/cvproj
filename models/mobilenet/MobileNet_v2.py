# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MobileNet_v2.py
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
from pathlib import Path
ROOT = str(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torchvision.models import (
    # mobilenetv2,
    mobilenet_v2,
    # MobileNetV2,
    MobileNet_V2_Weights,
    get_model_builder,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


net = mobilenet_v2(
    weights = MobileNet_V2_Weights.DEFAULT,
    progress = False,
)
net.eval()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
