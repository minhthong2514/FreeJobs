# Exporting model to engine for using tensorRT

## Method 1: Follow the constructions in tensorrtx repo (Recommended for Jetson Nano with python3.6)
You can find all information in this link:
```bash
https://github.com/wang-xinyu/tensorrtx/blob/master/yolov5/README.md
```
### Generate model pt to wts (weights):
```bash
python gen_wts.py -w farmbot_seg_model.pt -o farmbot_seg_model.wts -t seg
```
**Note:** Type seg is segmentation, default is detection. If you reached `Segmentation fault (Core dumped)`, You should switch to another device.

## Method 2: Converting step by step from pt to engine (Recommended for Jetson which can install ultralyctics)
### Exporting pt format to onnx format
This step you can export in other device, as sample is kaggle or google colab.

### Exporting onnx format to engine format
- This is importing step for deploying on jetson nano, you should use 16fp quantization for jetson nano.
- You can using this bash:
```bash
/usr/src/tensorrt/bin/trtexec --onnx=farmbot_seg_model.onnx --saveEngine=farmbot_seg_model.engine --fp16 --workspace=1024 --verbose
```
The first part is your trtexec path. The '--fp16' for quantizating (FP16) and the '--verbose' is used for printing detail on log, you can not use if you dont need it.

**IMPORTANT**: Exporting to engine must do on your main device.