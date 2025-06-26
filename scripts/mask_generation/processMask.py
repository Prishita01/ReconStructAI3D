import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator
from tqdm import tqdm
from utils import mkdir_safely  # assuming you have this like in your original code

def segment_sam(config):
    # Setup directories
    workdir = config['object_dir']
    input_dir = os.path.join(workdir, config['frames_dir'])
    mask_dir = os.path.join(workdir, config['mask_dir'])

    logger = config['logger']

    mkdir_safely(mask_dir)

    # Model setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "scripts/weights/sam_vit_h_4b8939.pth"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # Process images
    image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    for file_name in tqdm(image_list, desc="Segmenting"):
        image_path = os.path.join(input_dir, file_name)
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Inference
        predictor.set_image(image_rgb)

        height, width, _ = image_rgb.shape
        input_point = np.array([[width // 2, height // 2]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        best_mask = masks[np.argmax(scores)]
        binary_mask = (best_mask * 255).astype(np.uint8)

        # Save binary mask
        mask_path = os.path.join(mask_dir, file_name.replace('.jpg', '.png'))
        cv2.imwrite(mask_path, binary_mask)

    logger.info(f" Masks are saved to: {mask_dir}")
