
<details><summary>目录</summary><p>

- [Image Segmentation](#image-segmentation)
- [Object Detection](#object-detection)
    - [Object Detection](#object-detection-1)
    - [Face Detection](#face-detection)
    - [Text Detection OCR](#text-detection-ocr)
        - [模型](#模型)
        - [Keras OCR](#keras-ocr)
    - [YOLO](#yolo)
        - [文档](#文档)
        - [模型数据](#模型数据)
        - [模型训练](#模型训练)
        - [模型部署](#模型部署)
- [参考](#参考)
</p></details><p></p>

# Image Segmentation

* instance segmentation
    - faster-r-cnn
    - mask-r-cnn
* semantic segmentation
    - DeepLab
    - fast-fcn
    - LRASPP

# Object Detection

## Object Detection

* keypoint detection
* object detection
* object tracking
* ocr

## Face Detection

* face recognition
* face detection
* face filter
* face unlock

## Text Detection OCR

### 模型

* [CRNN](https://github.com/janzd/CRNN)
* [CRAFT](https://github.com/clovaai/CRAFT-pytorch)
* Differentiable Binarization

### Keras OCR

* [keras-ocr GitHub](https://github.com/faustomorales/keras-ocr)
* [keras-ocr Doc](https://keras-ocr.readthedocs.io/en/latest/examples/fine_tuning_detector.html)

## YOLO

### 文档

* [ultralytics doc](https://docs.ultralytics.com/)
* [ultralytics hub](https://hub.ultralytics.com/home)
* [ultralytics hub GitHub](https://github.com/ultralytics/hub/tree/master/coco6)

### 模型数据

* 数据标注
    - [make sense](https://www.makesense.ai/)
    - [roboflow](https://app.roboflow.com/)

### 模型训练

* 模型
    - [Yolo3 GitHub](https://github.com/ultralytics/yolov3)
    - [Yolo5 GitHub](https://github.com/ultralytics/yolov5)
    - [Yolo6 GitHub](https://github.com/ultralytics/yolov6)
    - [meituan Yolo6 GitHub](https://github.com/meituan/YOLOv6)
    - [Yolo7 GitHub](https://github.com/WongKinYiu/yolov7)
    - [Yolo8 GitHub](https://github.com/ultralytics/ultralytics)
* Notebooks
    - [Paperspace]()
* Logger
    - [Clear ML]()
    - [comet](https://www.comet.com/site/?ref=yolov5&utm_source=yolov5&utm_medium=affilliate&utm_campaign=yolov5_comet_integration)

### 模型部署

* Platforms
    - [Neural Magic]()
* Exports
    - [PyTorch](https://pytorch.org/)
        - `yolov5.pt`
    - [TorchScript](https://pytorch.org/docs/stable/jit.html)
        - `yolov5.torchscript`
    - [ONNX](https://onnx.ai/)
        - `yolov5.onnx`
    - [OpenVINO](https://docs.openvino.ai/latest/home.html)
        - `yolov5_openvino_model/`
    - [TensorRT](https://developer.nvidia.com/tensorrt)
        - `yolov5.engine`
    - [CoreML](https://github.com/apple/coremltools)
        - `yolov5.mlmodel`
    - [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model?hl=zh-cn)
        - `yolov5_saved_model/`
    - [TensorFlow GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph)
        - `yolov5.pb`
    - [TensorFlow Lite](https://www.tensorflow.org/lite?hl=zh-cn)
        - `yolov5.tflite`
    - [TensorFlow Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)
        - `yolov5_edgetpu.tflite`
    - [TensorFlow.js](https://www.tensorflow.org/js?hl=zh-cn)
        - `yolov5_web_model/`
    - [PaddlePaddle](https://github.com/PaddlePaddle)
        - `yolov5_paddle_model/`

# 参考
