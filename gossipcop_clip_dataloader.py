# ./mm/gossipcop_clip_dataloader.py

import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from torchvision import transforms
import os
import numpy as np
from PIL import Image, ImageFile
import logging
from typing import Optional

from transformers import BertTokenizer, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True # 允许加载截断的图像文件
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image_mae(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        try:
            interpolation_mode = transforms.InterpolationMode.BICUBIC
        except AttributeError:
             interpolation_mode = Image.BICUBIC
        data_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=interpolation_mode),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return data_transforms(img)
    except FileNotFoundError:
        logger.debug(f"MAE Image not found: {image_path}. Returning None.") # 改为 debug 减少日志噪音
        return None
    except Exception as e:
        logger.warning(f"Error processing MAE image {image_path}: {e}. Returning None.")
        return None

def preprocess_image_clip(image_path, clip_processor):
    try:
        img = Image.open(image_path).convert('RGB')
        processed = clip_processor(images=img, return_tensors="pt", truncation=True) # 添加 truncation
        return processed.pixel_values.squeeze(0)
    except FileNotFoundError:
        logger.debug(f"CLIP Image not found: {image_path}. Returning None.") # 改为 debug
        return None
    except Exception as e:
        logger.warning(f"Error processing CLIP image {image_path}: {e}. Returning None.")
        return None

def word2input(texts, vocab_file_or_model_id, max_len):
    try:
        tokenizer = BertTokenizer.from_pretrained(vocab_file_or_model_id, do_lower_case=True)
        logger.info(f"BERT Tokenizer loaded using: {vocab_file_or_model_id}")
    except Exception as e:
        logger.error(f"Failed to load BERT tokenizer from {vocab_file_or_model_id}: {e}")
        raise

    all_input_ids = []
    all_attention_masks = []
    for text in texts:
        text = str(text) if text is not None else ""
        encoded_dict = tokenizer.encode_plus(
            text,
            max_length=max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        all_input_ids.append(encoded_dict['input_ids'].squeeze(0))
        all_attention_masks.append(encoded_dict['attention_mask'].squeeze(0))
    return torch.stack(all_input_ids), torch.stack(all_attention_masks)


def _init_fn(worker_id):
    np.random.seed(2025 + worker_id)

class bert_data():
    def __init__(self,
                 max_len: int,
                 batch_size: int,
                 vocab_file: str, 
                 num_workers: int = 2,
                 data_dir: str = './gossipcop/',
                 image_dir_base: str = './gossipcop/',
                 clip_model_path: str = "./pretrained_model/clip-vit-base-patch16"):

        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file 
        self.data_dir = data_dir
        self.image_dir_base = image_dir_base
        self.category_dict = {"gossip": 0} 
        self.dummy_category_index = self.category_dict["gossip"]

        self.clip_model_path = clip_model_path
        logger.info(f"Loading CLIP Processor from local path: {self.clip_model_path}")
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(
                self.clip_model_path,
                local_files_only=True 
            )
            logger.info("CLIP Processor loaded successfully from local files.")
        except Exception as e:
            logger.error(f"Failed to load CLIP Processor from {self.clip_model_path}. "
                         f"Ensure directory exists and contains necessary files: {e}")
            raise

    def load_data(self,
                  csv_filename: str,
                  image_folder_suffix: str,
                  shuffle: bool,
                  use_preprocessed_images: bool = False, # 这个参数现在主要控制图像PKL
                  mae_pkl_path: Optional[str] = None,
                  clip_pkl_path: Optional[str] = None,
                  text_pkl_path: Optional[str] = None): # 文本PKL仍然是可选的

        csv_path = os.path.join(self.data_dir, csv_filename)
        image_dir = os.path.join(self.image_dir_base, image_folder_suffix)
        logger.info(f"Loading data from CSV: {csv_path}")
        logger.info(f"Image directory (for on-the-fly if needed): {image_dir}")

        try:
            data_df = pd.read_csv(csv_path, encoding='utf-8')
            required_cols = ['post_text', 'label', 'image_id'] # 'id' can be a fallback for image_id
            if 'image_id' not in data_df.columns and 'id' in data_df.columns:
                data_df.rename(columns={'id': 'image_id'}, inplace=True)
            
            for col in required_cols:
                if col not in data_df.columns:
                    logger.error(f"Required column '{col}' not found in {csv_path}")
                    return None
            data_df['post_text'] = data_df['post_text'].fillna('')
            logger.info(f"Loaded {len(data_df)} rows from CSV.")
        except FileNotFoundError:
             logger.error(f"CSV file not found: {csv_path}")
             return None
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            return None

        bert_token_ids_list = []
        bert_masks_list = []
        labels_list = []
        categories_list = []
        mae_images_list = []
        clip_pixel_values_list = []
        clip_input_ids_list = []
        clip_attention_mask_list = []

        preprocessed_mae_images = None
        preprocessed_clip_images = None
        preprocessed_text_data = None
        
        mae_loaded_from_pkl = False
        clip_img_loaded_from_pkl = False
        text_loaded_from_pkl = False
        expected_len = len(data_df)

        if use_preprocessed_images: # This flag now primarily gates image PKL loading
            logger.info("Attempting to load preprocessed IMAGE data from PKL files...")

            def load_img_pkl(path, name):
                data = None
                loaded_flag = False
                if path and os.path.exists(path):
                    try:
                        with open(path, 'rb') as f: data = pickle.load(f)
                        if len(data) == expected_len:
                            logger.info(f"Successfully loaded {name} PKL: {path} with {len(data)} items.")
                            loaded_flag = True
                        else:
                            logger.warning(f"Length mismatch for {name} PKL: {path}. Expected {expected_len}, got {len(data)}. Ignoring this PKL.")
                            data = None
                    except Exception as e:
                        logger.error(f"Error loading {name} PKL {path}: {e}. Ignoring this PKL.")
                        data = None
                else:
                    logger.warning(f"{name} PKL path not provided or file not found: {path}.")
                return data, loaded_flag

            preprocessed_mae_images, mae_loaded_from_pkl = load_img_pkl(mae_pkl_path, "MAE Image")
            preprocessed_clip_images, clip_img_loaded_from_pkl = load_img_pkl(clip_pkl_path, "CLIP Image")
        
        # 尝试加载文本PKL (独立于 use_preprocessed_images，因为它控制图像)
        if text_pkl_path and os.path.exists(text_pkl_path):
            try:
                with open(text_pkl_path, 'rb') as f: temp_text_data = pickle.load(f)
                required_keys = ['bert_token_ids', 'bert_masks', 'labels', 'categories', 'clip_input_ids', 'clip_attention_mask']
                if all(key in temp_text_data for key in required_keys) and \
                   all(len(temp_text_data[key]) == expected_len for key in required_keys):
                    preprocessed_text_data = temp_text_data
                    text_loaded_from_pkl = True
                    logger.info(f"Successfully loaded Text PKL: {text_pkl_path} with valid structure and length.")
                else:
                    logger.warning(f"Text PKL {text_pkl_path} missing keys or has length mismatch. Will process text on-the-fly.")
            except Exception as e:
                logger.error(f"Error loading or validating Text PKL {text_pkl_path}: {e}. Will process text on-the-fly.")
        else:
            logger.info(f"Text PKL path not provided or file not found: {text_pkl_path}. Text will be processed on-the-fly.")

        # 根据是否从PKL加载文本来决定如何处理文本
        if text_loaded_from_pkl:
            logger.info("Using preprocessed text data from PKL.")
            bert_token_ids_all = preprocessed_text_data['bert_token_ids']
            bert_masks_all = preprocessed_text_data['bert_masks']
            labels_all = preprocessed_text_data['labels'] # 这些是列表
            categories_all = preprocessed_text_data['categories'] # 这些是列表
            clip_input_ids_all = preprocessed_text_data['clip_input_ids']
            clip_attention_mask_all = preprocessed_text_data['clip_attention_mask']
        else:
            logger.info("Processing text data on-the-fly...")
            all_texts = data_df['post_text'].tolist()
            labels_all = data_df['label'].astype(int).tolist() # 从DataFrame获取标签
            categories_all = [self.dummy_category_index] * len(data_df) # 创建dummy类别列表

            logger.info("Tokenizing all texts for BERT...")
            bert_token_ids_all, bert_masks_all = word2input(all_texts, self.vocab_file, self.max_len)
            
            logger.info("Tokenizing all texts for CLIP...")
            clip_max_len_actual = 77 
            if hasattr(self.clip_processor, 'tokenizer') and hasattr(self.clip_processor.tokenizer, 'model_max_length'):
                clip_max_len_actual = self.clip_processor.tokenizer.model_max_length
            elif hasattr(self.clip_processor, 'model_max_length'):
                 clip_max_len_actual = self.clip_processor.model_max_length
            elif hasattr(self.clip_processor, 'tokenizer') and hasattr(self.clip_processor.tokenizer, 'context_length'):
                 clip_max_len_actual = self.clip_processor.tokenizer.context_length
            else:
                logger.warning(f"CLIPProcessor does not have model_max_length. Using default: {clip_max_len_actual} for CLIP text.")

            clip_tokenizer_outputs = self.clip_processor(
                text=all_texts, return_tensors="pt", padding='max_length', truncation=True, max_length=clip_max_len_actual
            )
            clip_input_ids_all = clip_tokenizer_outputs['input_ids']
            clip_attention_mask_all = clip_tokenizer_outputs['attention_mask']
            logger.info("Text tokenization complete.")

        # 迭代处理每一行数据
        processed_count = 0
        skipped_due_to_missing_image_pkl = 0
        skipped_due_to_image_processing_failure = 0
        
        for index in range(len(data_df)):
            # 获取文本特征 (来自PKL或实时处理的结果)
            current_bert_token_ids = bert_token_ids_all[index]
            current_bert_masks = bert_masks_all[index]
            current_label = labels_all[index]
            current_category = categories_all[index]
            current_clip_input_ids = clip_input_ids_all[index]
            current_clip_attention_mask = clip_attention_mask_all[index]

            mae_img_tensor = None
            clip_pixel_values = None
            
            # 尝试从PKL加载图像特征
            if mae_loaded_from_pkl and preprocessed_mae_images is not None and index < len(preprocessed_mae_images):
                mae_img_tensor = preprocessed_mae_images[index]
            if clip_img_loaded_from_pkl and preprocessed_clip_images is not None and index < len(preprocessed_clip_images):
                clip_pixel_values = preprocessed_clip_images[index]

            # 如果PKL中的图像数据是None (表示预处理失败或原始图像缺失)，则该样本跳过
            if (mae_loaded_from_pkl and mae_img_tensor is None) or \
               (clip_img_loaded_from_pkl and clip_pixel_values is None):
                logger.debug(f"Skipping sample index {index} (Image ID: {data_df.iloc[index]['image_id']}) because preprocessed image data in PKL is None.")
                skipped_due_to_missing_image_pkl += 1
                continue
            
            # 如果没有从PKL加载图像，则实时处理
            if not mae_loaded_from_pkl or not clip_img_loaded_from_pkl:
                image_id = str(data_df.iloc[index]['image_id'])
                possible_paths = [
                     os.path.join(image_dir, image_id), os.path.join(image_dir, f"{image_id}.jpg"),
                     os.path.join(image_dir, f"{image_id}.png"), os.path.join(image_dir, f"{image_id}.jpeg"),
                     os.path.join(image_dir, f"{image_id}.webp")
                ]
                image_path = next((p for p in possible_paths if os.path.exists(p)), None)

                if image_path:
                    if not mae_loaded_from_pkl:
                        mae_img_tensor = preprocess_image_mae(image_path)
                    if not clip_img_loaded_from_pkl:
                        clip_pixel_values = preprocess_image_clip(image_path, self.clip_processor)
                else: # 图像文件未找到
                    logger.debug(f"Skipping sample index {index} (Image ID: {image_id}) as image file not found for on-the-fly processing.")
                    skipped_due_to_image_processing_failure +=1
                    continue # 跳过此样本

                # 如果实时处理后仍然有一个图像是None，则跳过
                if mae_img_tensor is None or clip_pixel_values is None:
                    logger.debug(f"Skipping sample index {index} (Image ID: {image_id}) due to on-the-fly image processing failure.")
                    skipped_due_to_image_processing_failure +=1
                    continue
            
            # 至此，所有特征都已获取或处理
            bert_token_ids_list.append(current_bert_token_ids)
            bert_masks_list.append(current_bert_masks)
            labels_list.append(current_label)
            categories_list.append(current_category)
            mae_images_list.append(mae_img_tensor)
            clip_pixel_values_list.append(clip_pixel_values)
            clip_input_ids_list.append(current_clip_input_ids)
            clip_attention_mask_list.append(current_clip_attention_mask)
            processed_count += 1
            
            if (index + 1) % 2000 == 0: #减少日志频率
                 logger.info(f"Aggregated {index + 1}/{len(data_df)} samples...")

        logger.info(f"Data aggregation complete. Total valid samples: {processed_count}.")
        if skipped_due_to_missing_image_pkl > 0:
            logger.warning(f"Skipped {skipped_due_to_missing_image_pkl} samples due to None in image PKLs.")
        if skipped_due_to_image_processing_failure > 0:
            logger.warning(f"Skipped {skipped_due_to_image_processing_failure} samples due to on-the-fly image processing failures or file not found.")


        if not bert_token_ids_list:
            logger.error("No valid data loaded or processed. Cannot create DataLoader.")
            return None

        logger.info("Converting aggregated data to Tensors...")
        try:
            bert_token_ids_tensor = torch.stack(bert_token_ids_list)
            bert_masks_tensor = torch.stack(bert_masks_list)
            label_tensor = torch.tensor(labels_list, dtype=torch.long)
            category_tensor = torch.tensor(categories_list, dtype=torch.long)
            image_mae_tensor = torch.stack(mae_images_list)
            clip_pixel_values_tensor = torch.stack(clip_pixel_values_list)
            clip_input_ids_tensor = torch.stack(clip_input_ids_list)
            clip_attention_mask_tensor = torch.stack(clip_attention_mask_list)
            logger.info("Tensor conversion successful.")
        except Exception as e:
            logger.error(f"Error stacking tensors: {e}. Check if all elements in lists are valid tensors of consistent shape.")
            logger.error(f"Shapes: BERT_IDs={len(bert_token_ids_list)}, MAE_Imgs={len(mae_images_list)}, CLIP_Imgs={len(clip_pixel_values_list)}")
            # 打印一些样本的形状以帮助调试
            if len(mae_images_list) > 0: logger.error(f"Sample MAE image shape: {mae_images_list[0].shape if isinstance(mae_images_list[0], torch.Tensor) else type(mae_images_list[0])}")
            if len(clip_pixel_values_list) > 0: logger.error(f"Sample CLIP image shape: {clip_pixel_values_list[0].shape if isinstance(clip_pixel_values_list[0], torch.Tensor) else type(clip_pixel_values_list[0])}")
            return None

        datasets = TensorDataset(
            bert_token_ids_tensor, bert_masks_tensor, label_tensor, category_tensor,
            image_mae_tensor, clip_pixel_values_tensor, clip_input_ids_tensor, clip_attention_mask_tensor
        )
        logger.info(f"TensorDataset created with {len(datasets)} samples.")

        dataloader = DataLoader(
            dataset=datasets, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, shuffle=shuffle, worker_init_fn=_init_fn
        )
        logger.info(f"DataLoader created with {len(dataloader)} batches (Batch size: {self.batch_size}).")

        return dataloader