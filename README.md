# anomaly_detection

## 環境建築
The code has been tested with these PC environments:
- Ubuntu 20.04.6 LTS
- CUDA Version: 11.7
- NVIDIA Corporation GA102GL [RTX A5000]
1. Create a new conda environment:

   ```bash
   conda create -n cheating-detection-server python=3.9
   conda activate cheating-detection-server
   ```
   
2. Install Pytorch (1.7.1 is recommended).
   ```bash
   pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   ```

4. Install MMDetection. 

   * Install [MMCV-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) first. 1.4.8 is recommended.
     ```bash
     pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
     ```

   * ```bash
     cd cheating-detection-app-server
     pip install -v -e .
     ```
## 重みファイル
1. Install gdown for weights downloading

   ```bash
   pip install gdown
   ```
2. Download weights file for gaze estimation:

   ```bash
   mkdir ckpts && cd ckpts
   # gaze estimtion
   gdown 'https://drive.google.com/uc?id=1ru0xhuB5N9kwvN9XLvZMQvVSfOgtbxmq'
   # face detection
   gdown 'https://drive.google.com/uc?id=1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb'
   ```
   
3. Download weights file for pose estimation:
   ```bash
   cd OpenPoseNet
   mkdir weights && cd weights
   wget -O pose_model_scratch.pth 'https://www.dropbox.com/s/5v654d2u65fuvyr/pose_model_scratch.pth?e=1&dl=1'
   ```
If you encounter difficulties in running command for downloading weight, please download the checkpoint for the model from [gaze estimation](https://drive.google.com/drive/folders/1OX_nuxXYTH5i8E11UCyEcAsp6ExHDMra), [face detection](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view) and [pose estimation](https://www.dropbox.com/s/5v654d2u65fuvyr/pose_model_scratch.pth?e=1&dl=0)
## 実行できるコマンド
1. モデルの動作確認

   ```bash
   python gaze_estimation.py
   python detect_human.py
   ```
2. 不正行為の検知

   ```bash
   python pose_estimation.py
   ```
   
## 謝辞
This code is inspired by [MCGaze](https://github.com/zgchen33/MCGaze/tree/master), [OpenPoseNet](https://github.com/YutaroOgawa/pytorch_advanced/blob/master/4_pose_estimation/4-7_OpenPose_inference.ipynb), [MMDetection](https://github.com/open-mmlab/mmdetection) and [FastRCNN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html). Thanks for their great contributions to the computer vision community.
