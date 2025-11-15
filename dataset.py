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

class MonkeyDataset(Dataset):
    def __init__(self, data_root_dir, split='train'):
        self.data_root_dir = data_root_dir
        self.split = split
        self.patch_size = PATCH_SIZE
        
        self.pas_cpg_dir = os.path.join(data_root_dir, 'images', 'pas-cpg')
        self.xml_dir = os.path.join(data_root_dir, 'annotations', 'xml')
        self.mask_dir = os.path.join(data_root_dir, 'images', 'tissue-masks')
        
        self.annotations = self._load_all_annotations(self.xml_dir) 
        self.roi_bounds = self._load_roi_bounds(self.mask_dir)

        self.samples = self._create_patch_samples(self.roi_bounds)
        self.transform = self._get_augmentations(split)
        
    def _load_all_annotations(self, xml_dir):
        all_annotations = {}
        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
        
        print(f"Loading annotations from {len(xml_files)} XML files...")

        for filename in tqdm(xml_files):
            xml_path = os.path.join(xml_dir, filename)
            slide_id = os.path.splitext(filename)[0]
            
            annotations_list = parse_xml_annotations(xml_path)
            
            if annotations_list:
                all_annotations[slide_id] = annotations_list
                
        print(f"Successfully loaded annotations for {len(all_annotations)} slides.")
        
        return all_annotations
    
    def _load_roi_bounds(self, tissue_mask_dir):
        roi_bounds = {}
        mask_files = [f for f in os.listdir(tissue_mask_dir) if f.endswith('.tif')]
        
        print(f"Calculating ROI bounds from {len(mask_files)} TIF masks...")

        for mask_filename in tqdm(mask_files):
            mask_path = os.path.join(tissue_mask_dir, mask_filename)
            
            match = re.match(r'(.+)_mask\.tif', mask_filename)

            if not match: continue
            slide_id = match.group(1) 
            
            mask_slide = None

            try:
                mask_slide = openslide.OpenSlide(mask_path)

                level_to_read = min(3, mask_slide.level_count - 1) 
                mask_dimensions = mask_slide.level_dimensions[level_to_read]
                
                mask_patch = mask_slide.read_region((0, 0), level_to_read, mask_dimensions)
                mask_data = np.array(mask_patch.convert("L")) 
                
                y_coords, x_coords = np.where(mask_data > 0) 
                
                if len(x_coords) > 0:
                    xmin_low, ymin_low = x_coords.min(), y_coords.min()
                    xmax_low, ymax_low = x_coords.max(), y_coords.max()
                    
                    downsample_factor = mask_slide.level_downsamples[level_to_read]
                    
                    xmin = int(xmin_low * downsample_factor)
                    ymin = int(ymin_low * downsample_factor)
                    xmax = int(xmax_low * downsample_factor)
                    ymax = int(ymax_low * downsample_factor)
                    
                    roi_bounds[slide_id] = {'xmin': xmin, 'ymin': ymin,
                                            'xmax': xmax, 'ymax': ymax}
                
            except Exception as e:
                print(f"Warning: Error processing mask {mask_filename}. Skipping. Error: {e}")

            finally:
                if mask_slide is not None:
                    mask_slide.close()
                
        print(f"Successfully computed ROI bounds for {len(roi_bounds)} slides from TIF masks.")

        return roi_bounds