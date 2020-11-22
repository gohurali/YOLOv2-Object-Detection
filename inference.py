import numpy as np
import yaml
import cv2
import os
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import argparse
from utils.data_utils import Dataset
from utils.data_utils import DataPrepper
from models.darknet import DarkNet19
from models.yolo import YOLOv2
from models.losses import YOLO_SSE
from utils.dataset import *

class Inferencer:
    def __init__(self):
        pass

    def open_model(self):
        pass

    def inference(self):
        pass

def main():
    pass

if __name__ == "__main__":
    main()