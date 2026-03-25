# gossipcop_clip_data_pre.py
import pandas as pd
import os
import numpy as np
import torch
import pickle
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
import logging
from tqdm import tqdm # For progress bar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load and preprocess a single image (CLIP style) - same as in dataloader
def preprocess_image_clip(image_path, clip_preprocess):
    try:
        img = Image.open(image_path).convert('RGB')
        # Important: Move to CPU before saving in PKL, move to GPU in training loop
        return clip_preprocess(img).squeeze(0).cpu() # Remove batch dim, move to CPU
    except FileNotFoundError:
        logger.warning(f"CLIP Image not found: {image_path}. Returning None.")
        return None
    except Exception as e:
        logger.error(f"Error processing CLIP image {image_path}: {e}. Returning None.")
        return None

def create_clip_pkl(csv_path, image_dir, output_pkl_path, clip_preprocess):
    logger.info(f"Processing CSV: {csv_path}")
    logger.info(f"Reading images from: {image_dir}")
    logger.info(f"Outputting to: {output_pkl_path}")

    try:
        data_df = pd.read_csv(csv_path, encoding='utf-8')
        logger.info(f"Read {len(data_df)} rows.")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        return
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
        return

    ordered_images_clip = []
    processed_count = 0
    missing_count = 0

    for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Preprocessing CLIP Images"):
        image_id = str(row['image_id'])
        possible_paths = [
            os.path.join(image_dir, image_id),
            os.path.join(image_dir, image_id + '.jpg'),
            os.path.join(image_dir, image_id + '.png'),
        ]
        image_path = None
        for p in possible_paths:
            if os.path.exists(p):
                image_path = p
                break

        if image_path:
            img_tensor = preprocess_image_clip(image_path, clip_preprocess)
            if img_tensor is not None:
                ordered_images_clip.append(img_tensor)
                processed_count += 1
            else:
                ordered_images_clip.append(None) # Mark as missing
                missing_count += 1
        else:
            ordered_images_clip.append(None) # Mark as missing
            missing_count += 1
            if index < 10: # Log only a few missing examples
                 logger.warning(f"Row {index}: CLIP Image file not found for ID {image_id}")


    logger.info(f"Finished. Processed: {processed_count}, Missing/Failed: {missing_count}")

    # Saving list potentially containing None values
    try:
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(ordered_images_clip, f)
        logger.info(f"Successfully saved {len(ordered_images_clip)} CLIP image tensors/placeholders to {output_pkl_path}")
    except Exception as e:
        logger.error(f"Failed to save PKL file {output_pkl_path}: {e}")

# --- Configuration ---
GOSSIPCOP_DIR = './gossipcop/' # Adjust path if needed
TRAIN_CSV = os.path.join(GOSSIPCOP_DIR, 'gossip_train.csv')
TEST_CSV = os.path.join(GOSSIPCOP_DIR, 'gossip_test.csv')
TRAIN_IMG_DIR = os.path.join(GOSSIPCOP_DIR, 'gossip_train')
TEST_IMG_DIR = os.path.join(GOSSIPCOP_DIR, 'gossip_test')
TRAIN_CLIP_PKL = os.path.join(GOSSIPCOP_DIR, 'gossip_train_clip.pkl')
TEST_CLIP_PKL = os.path.join(GOSSIPCOP_DIR, 'gossip_test_clip.pkl')

# --- Initialize CLIP for preprocessing ---
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device} for CLIP loading")
try:
    # Load CLIP model once to get the preprocess function
    _, clip_preprocess_fn = load_from_name("ViT-B-16", device=device, download_root='./')
    logger.info("CLIP preprocess function ready.")

    # --- Run Preprocessing ---
    create_clip_pkl(TRAIN_CSV, TRAIN_IMG_DIR, TRAIN_CLIP_PKL, clip_preprocess_fn)
    create_clip_pkl(TEST_CSV, TEST_IMG_DIR, TEST_CLIP_PKL, clip_preprocess_fn)

except Exception as e:
    logger.error(f"Failed to load CLIP model or run preprocessing: {e}")