import astra
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from model import UNet
from model_mutihead import UNet3Head

# import 3 model individually, train them with 3 individual loss
from models_3_path.model1 import UNet3_1
from models_3_path.model2 import UNet3_2
from models_3_path.model3 import UNet3_3


from projectionDataloader import ProjectionDataset_16, ProjectionDataset_32, ProjectionDataset_16_32, ProjectionDataset_2_4, ProjectionDataset_4_8
import onnx
import onnx
import onnx.utils
import onnx.version_converter
import os
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter

def infer_epoch():
    