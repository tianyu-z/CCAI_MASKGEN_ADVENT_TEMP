# CCAI_MASKGEN_ADVENT_TEMP
## What needs to be done before run the code?
1. Create a new subfolder under the root named **pretrained_models**  
2. Download the [pretrained DeepLabv2 model](https://github.com/valeoai/ADVENT/releases/download/v0.1/DeepLab_resnet_pretrained_imagenet.pth) and save it in the folder  
3. Change the comet_ml work_space and user in the config files  

## How to run the code?
```bash
pip install .
python train_CCAI.py
```
