import torch
from ultralytics import YOLO
# YOLO("yolov8n.pt")

# import requests
# import zipfile
import os
# import glob
# import cv2
# import matplotlib.pyplot as plt
# import random
# import numpy as np
import yaml
import subprocess


if __name__ == '__main__':
    # Get the path of the current working directory
    cwd = os.getcwd()
    print(cwd)

    EPOCHS = 40
    BATCH = 24
    IM_SIZE = 640

    # command = 'yolo task=detect mode=train model=yolov8n.pt imgsz='+str(IM_SIZE)+' data=waste_instance.yaml batch='+str(BATCH)+' epochs='+str(EPOCHS)+' name=yolov8n exist_ok=True amp=False'
    # result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # print(result.stdout)

    torch.cuda.empty_cache()

    model = YOLO('yolov8n.pt')
    result = model.train(data=cwd+'/dataset/waste_instance.yaml', epochs=EPOCHS, device='cuda')