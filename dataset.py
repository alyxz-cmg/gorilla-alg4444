import torch
from torch.utils.data import Dataset
import numpy as np
import openslide
from PIL import Image
from utils import create_heatmap, PATCH_SIZE, parse_xml_annotations 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import os
import json
import re 
from tqdm import tqdm 