# gossipcop_data_pre.py
import pandas as pd
import os
import numpy as np
import torch
import pickle
from PIL import Image
from torchvision import transforms
import logging
from tqdm import tqdm # For progress bar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load and preprocess a single image (MAE style) - same as in dataloader
def preprocess_image_mae(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        data_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return data_transforms(img)
    except FileNotFoundError:
        logger.warning(f"MAE Image not found: {image_path}. Returning None.")
        return None
    except Exception as e:
        logger.error(f"Error processing MAE image {image_path}: {e}. Returning None.")
        return None

def create_mae_pkl(csv_path, image_dir, output_pkl_path):
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

    ordered_images_mae = []
    processed_count = 0
    missing_count = 0

    for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Preprocessing MAE Images"):
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
            img_tensor = preprocess_image_mae(image_path)
            if img_tensor is not None:
                ordered_images_mae.append(img_tensor)
                processed_count += 1
            else:
                # Append None or a default tensor if processing failed but row needs alignment
                ordered_images_mae.append(None) # Mark as missing for dataloader check
                missing_count += 1
        else:
            # Append None or a default tensor if image file missing but row needs alignment
            ordered_images_mae.append(None) # Mark as missing for dataloader check
            missing_count += 1
            if index < 10: # Log only a few missing examples
                 logger.warning(f"Row {index}: MAE Image file not found for ID {image_id}")

    logger.info(f"Finished. Processed: {processed_count}, Missing/Failed: {missing_count}")

    # Note: Saving a list containing None values. The dataloader needs to handle this.
    # Alternatively, filter out rows with missing images *before* saving,
    # but this requires saving the filtered DataFrame/indices too.
    # Saving with None keeps alignment with the original CSV.
    try:
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(ordered_images_mae, f)
        logger.info(f"Successfully saved {len(ordered_images_mae)} MAE image tensors/placeholders to {output_pkl_path}")
    except Exception as e:
        logger.error(f"Failed to save PKL file {output_pkl_path}: {e}")

# --- Configuration ---
GOSSIPCOP_DIR = './gossipcop/' # Adjust path if needed
TRAIN_CSV = os.path.join(GOSSIPCOP_DIR, 'gossip_train.csv')
TEST_CSV = os.path.join(GOSSIPCOP_DIR, 'gossip_test.csv')
TRAIN_IMG_DIR = os.path.join(GOSSIPCOP_DIR, 'gossip_train')
TEST_IMG_DIR = os.path.join(GOSSIPCOP_DIR, 'gossip_test')
TRAIN_MAE_PKL = os.path.join(GOSSIPCOP_DIR, 'gossip_train_mae.pkl')
TEST_MAE_PKL = os.path.join(GOSSIPCOP_DIR, 'gossip_test_mae.pkl')

# --- Run Preprocessing ---
create_mae_pkl(TRAIN_CSV, TRAIN_IMG_DIR, TRAIN_MAE_PKL)
create_mae_pkl(TEST_CSV, TEST_IMG_DIR, TEST_MAE_PKL)