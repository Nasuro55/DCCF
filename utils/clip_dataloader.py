# utils/clip_dataloader.py
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import pandas as pd
import pickle
import os
import numpy as np
import cn_clip.clip as clip # 导入cn_clip
import logging

logger = logging.getLogger(__name__)

def _init_fn(worker_id):
    np.random.seed(int(torch.initial_seed()) % (2**32) + worker_id)

class CustomDataset(Dataset):
    def __init__(self, content_ids, content_masks, labels, categories, images, clip_images, clip_text_ids):
        self.content_ids = content_ids
        self.content_masks = content_masks
        self.labels = labels
        self.categories = categories
        self.images = images
        self.clip_images = clip_images
        self.clip_text_ids = clip_text_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.content_ids[idx],
            self.content_masks[idx],
            self.labels[idx],
            self.categories[idx],
            self.images[idx],
            self.clip_images[idx],
            self.clip_text_ids[idx]
        )

class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2, clip_model_name="ViT-B-16", clip_download_root='./'):
        self.bert_max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.vocab_file) # vocab_file 是 BERT 的 vocab.txt 路径

        self.clip_tokenizer = clip.tokenize # 直接获取 tokenize 函数

        # 修正：不再尝试从模型对象加载 context_length，而是根据模型名称直接指定
        if clip_model_name == "ViT-B-16":
            self.clip_context_length = 52  # 中文 ViT-B-16 的已知上下文长度
        elif "RN50" in clip_model_name: # 如果支持其他模型，可以添加
            self.clip_context_length = 77 # 例如 RN50 在 cn_clip.tokenize 中默认为77
        else:
            # 对于未知模型，可以给一个警告并使用 cn_clip.tokenize 的默认值 (77)，
            # 或者抛出错误。鉴于本项目主要使用 ViT-B-16，我们将主要关注它。
            logger.warning(
                f"Context length for CLIP model '{clip_model_name}' is not explicitly set. "
                f"Defaulting to 77. For Chinese 'ViT-B-16', it should be 52."
            )
            self.clip_context_length = 77 # cn_clip.tokenize 的默认值

        logger.info(f"CN-CLIP tokenizer will use context length: {self.clip_context_length} for model {clip_model_name}")
        # 注意：之前加载 clip_model_for_tokenizer 并 del 的代码块已移除，因为不再需要仅为获取 context_length 而加载模型。

    def bert_word2input(self, texts):
        token_ids = []
        masks = []
        for i, text in enumerate(texts):
            encoded_dict = self.bert_tokenizer.encode_plus(
                                text,
                                add_special_tokens=True,
                                max_length=self.bert_max_len,
                                padding='max_length',
                                return_attention_mask=True,
                                truncation=True,
                                return_tensors='pt'
                           )
            token_ids.append(encoded_dict['input_ids'].squeeze(0))
            masks.append(encoded_dict['attention_mask'].squeeze(0))

        token_ids = torch.stack(token_ids, dim=0)
        masks = torch.stack(masks, dim=0)
        return token_ids, masks

    def clip_text_tokenize(self, texts):
        try:
            # 使用 self.clip_context_length
            tokenized_texts = self.clip_tokenizer(texts, context_length=self.clip_context_length)
            # tokenize 函数可能返回 (N, 1, context_length) 或 (N, context_length)
            # 如果是 (N, 1, L)，则 squeeze(1)；如果是 (N, L)，则保持不变
            if tokenized_texts.ndim == 3 and tokenized_texts.shape[1] == 1:
                return tokenized_texts.squeeze(1)
            return tokenized_texts
        except Exception as e:
            logger.error(f"Error during CLIP tokenization: {e}")
            problematic_texts = []
            for i, text_item in enumerate(texts):
                try:
                    self.clip_tokenizer(text_item, context_length=self.clip_context_length)
                except:
                    problematic_texts.append((i, text_item))
            if problematic_texts:
                logger.error(f"Problematic texts for CLIP tokenize: {problematic_texts}")
            raise

    def load_data(self, data_path, image_pkl_path, clip_image_pkl_path, shuffle, text_only=False):
        logger.info(f"Loading data from CSV: {data_path}")
        logger.info(f"Loading standard images from PKL: {image_pkl_path}")
        logger.info(f"Loading CLIP images from PKL: {clip_image_pkl_path}")

        if data_path.endswith(".xlsx"):
            try:
                data_df = pd.read_excel(data_path)
            except ImportError:
                logger.error("Failed to read .xlsx file. 'openpyxl' library might be missing. Try 'pip install openpyxl'.")
                raise
        else:
            data_df = pd.read_csv(data_path, encoding='utf-8')

        all_content = data_df['content'].astype(str).tolist()
        all_labels = torch.tensor(data_df['label'].astype(int).tolist(), dtype=torch.float)
        all_categories = torch.tensor([self.category_dict[str(c)] for c in data_df['category']], dtype=torch.long)

        logger.info("Tokenizing text for BERT...")
        all_content_ids, all_content_masks = self.bert_word2input(all_content)
        logger.info("Tokenizing text for CLIP...")
        all_clip_text_ids = self.clip_text_tokenize(all_content)

        logger.info(f"Loading image tensors from {image_pkl_path}")
        if not os.path.exists(image_pkl_path):
            raise FileNotFoundError(f"Image PKL file not found: {image_pkl_path}. Please run data_pre.py first.")
        with open(image_pkl_path, 'rb') as f:
            all_images = pickle.load(f)
        if not isinstance(all_images, torch.Tensor): # PKL可能存的是list of tensors
            all_images = torch.stack(all_images) if isinstance(all_images, list) and len(all_images) > 0 else torch.empty(0)


        logger.info(f"Loading CLIP image tensors from {clip_image_pkl_path}")
        if not os.path.exists(clip_image_pkl_path):
            raise FileNotFoundError(f"CLIP Image PKL file not found: {clip_image_pkl_path}. Please run clip_data_pre.py first.")
        with open(clip_image_pkl_path, 'rb') as f:
            all_clip_images = pickle.load(f)
        if not isinstance(all_clip_images, torch.Tensor): # PKL可能存的是list of tensors
            all_clip_images = torch.stack(all_clip_images) if isinstance(all_clip_images, list) and len(all_clip_images) > 0 else torch.empty(0)

        num_samples = len(all_labels)
        data_lengths = {
            "Labels": len(all_labels),
            "BERT content_ids": len(all_content_ids),
            "BERT content_masks": len(all_content_masks),
            "Categories": len(all_categories),
            "Standard Images": len(all_images) if all_images is not None else 0,
            "CLIP Images": len(all_clip_images) if all_clip_images is not None else 0,
            "CLIP text_ids": len(all_clip_text_ids)
        }

        if not (data_lengths["BERT content_ids"] == num_samples and \
                data_lengths["BERT content_masks"] == num_samples and \
                data_lengths["Categories"] == num_samples and \
                data_lengths["Standard Images"] == num_samples and \
                data_lengths["CLIP Images"] == num_samples and \
                data_lengths["CLIP text_ids"] == num_samples):
            error_msg = "Data length mismatch after loading and tokenization:\n"
            for name, length in data_lengths.items():
                error_msg += f"{name}: {length} (Expected: {num_samples})\n"
            error_msg += "Please check your CSV and PKL files for consistency, especially image IDs and filtering steps in preprocessing."
            logger.error(error_msg)
            raise ValueError(error_msg)

        dataset = CustomDataset(all_content_ids, all_content_masks, all_labels, all_categories, all_images, all_clip_images, all_clip_text_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, worker_init_fn=_init_fn, drop_last=True if shuffle else False)

        logger.info(f"DataLoader for {data_path} created with {len(dataset)} samples.")
        return dataloader