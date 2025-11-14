import torch
import numpy as np
import openslide
from skimage.feature import peak_local_max
import os
import json
from train import build_model
from utils import PATCH_SIZE 
from torchvision import transforms as T
from tqdm import tqdm