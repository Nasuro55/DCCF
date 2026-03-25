# import random    原版
# # import cv2
# import torch
# import torch.utils.data as data
# # import data.util as util
# # try:
# #     import util
# # except ImportError as e:
# #     print(f"无法导入 util: {e}.")
# #     util = None

# from PIL import Image, ImageFile
# import os
# import pandas as pd
# import numpy as np
# # from tqdm import tqdm # 可选
# import torchvision.transforms as transforms
# import logging
# # 导入 Tokenizer 和 Processor
# from transformers import BertTokenizer, CLIPProcessor

# ImageFile.LOAD_TRUNCATED_IMAGES = True
# logger = logging.getLogger(__name__)

# class FakeNet_dataset(data.Dataset):
#     # --- !!! 关键修改：调整 __init__ 参数顺序 !!! ---
#     def __init__(
#         self,
#         # --- 没有默认值的参数在前 ---
#         root_path,                 # Required, no default
#         bert_tokenizer_instance, # Required, no default (new)
#         clip_processor_instance, # Required, no default (new)
#         # --- 有默认值的参数在后 ---
#         dataset_name="gossip",
#         image_size=224,
#         is_train=True,
#         data_augment=False,
#         duplicate_fake_times=0,
#         bert_max_len=197,
#         clip_max_len=77
#     ):
#         # --- !!! 重要：确保这里不再尝试加载 Tokenizer/Processor !!! ---
#         # --- 因为它们现在作为参数传入了 ---

#         # --- 初始化接收的实例 ---
#         self.bert_tokenizer = bert_tokenizer_instance
#         self.clip_processor = clip_processor_instance
#         if self.bert_tokenizer is None:
#              logger.warning("传入的 BERT tokenizer 实例为 None。")
#         if self.clip_processor is None:
#              logger.warning("传入的 CLIP processor 实例为 None。")

#         # --- 其他初始化代码 (保持不变) ---
#         self.duplicate_fake_times = duplicate_fake_times
#         self.dataset_name = dataset_name.lower()
#         assert (
#             self.dataset_name == "politi" or self.dataset_name == "gossip"
#         ), "错误! 只支持 'gossip' 或 'politi'!"
#         super(FakeNet_dataset, self).__init__()

#         logger.info(f"FakeNet_dataset 初始化: dataset_name='{self.dataset_name}', is_train={is_train}")
#         logger.info(f"重复虚假(标签=0)样本次数: {self.duplicate_fake_times}")

#         self.is_train = is_train
#         self.root_path = root_path
#         self.image_size = image_size
#         self.bert_max_len = bert_max_len
#         self.clip_max_len = clip_max_len

#         self.label_dict = []

#         dataset_folder_path = self.root_path
#         if not os.path.isdir(dataset_folder_path):
#              logger.error(f"提供的数据集根目录不是有效目录: {self.root_path}")
#              raise FileNotFoundError(f"数据集根目录无效: {self.root_path}")

#         csv_file_name = f"gossip_{'train' if self.is_train else 'test'}.csv"
#         csv_path = os.path.join(dataset_folder_path, csv_file_name)
#         if not os.path.exists(csv_path):
#             logger.error(f"在 {dataset_folder_path} 中未找到 CSV 文件: {csv_file_name}")
#             raise FileNotFoundError(f"在 {dataset_folder_path} 中未找到 CSV 文件: {csv_file_name}")

#         logger.info(f"从 CSV 加载数据: {csv_path}")
#         try:
#             data_df = pd.read_csv(csv_path)
#             potential_text_cols = ['content', 'text', 'post_text']
#             fill_dict = {col: '' for col in potential_text_cols if col in data_df.columns}
#             data_df.fillna(fill_dict, inplace=True)
#         except Exception as e:
#             logger.error(f"读取 CSV 文件 {csv_path} 失败: {e}")
#             raise

#         content_col = None
#         if 'post_text' in data_df.columns: content_col = 'post_text'
#         elif 'content' in data_df.columns: content_col = 'content'
#         elif 'text' in data_df.columns: content_col = 'text'
#         else: raise ValueError(f"CSV文件 {csv_path} 缺少文本列。")
#         logger.info(f"使用的文本列: '{content_col}'")

#         image_id_col = None
#         if 'image_id' in data_df.columns: image_id_col = 'image_id'
#         elif 'image' in data_df.columns: image_id_col = 'image'
#         else: raise ValueError(f"CSV文件 {csv_path} 缺少图像ID列。")
#         logger.info(f"使用的图像ID列: '{image_id_col}'")

#         label_col = 'label'
#         if label_col not in data_df.columns: raise ValueError(f"CSV文件 {csv_path} 缺少标签列。")

#         # 计算 pos_weight (0=Fake, 1=Real)
#         fake_news_num = 0 # 标签 0
#         real_news_num = 0 # 标签 1
#         valid_labels_count = 0
#         if label_col in data_df.columns:
#              for label_val in data_df[label_col]:
#                  try:
#                      label_int = int(label_val)
#                      if label_int == 0: fake_news_num += 1
#                      elif label_int == 1: real_news_num += 1
#                      else: continue
#                      valid_labels_count += 1
#                  except (ValueError, TypeError): pass
#         if valid_labels_count == 0: raise ValueError("数据集中没有有效的标签。")

#         if real_news_num == 0: self.pos_weight = torch.tensor(1.0); logger.warning("数据集中没有真实新闻样本（标签=1）。pos_weight 设置为 1.0。")
#         else: self.pos_weight = torch.tensor(fake_news_num / real_news_num) # num_negative / num_positive
#         self.thresh = real_news_num / valid_labels_count if valid_labels_count > 0 else 0.5
#         logger.info(f"有效标签数: {valid_labels_count}, 虚假新闻数 (标签=0): {fake_news_num}, 真实新闻数 (标签=1): {real_news_num}")
#         logger.info(f"计算得到的 pos_weight (fake/real): {self.pos_weight:.4f}, thresh (real ratio): {self.thresh:.4f}")

#         # 构建 label_dict
#         skipped_num = 0
#         image_folder_name = f"{self.dataset_name}_{'train' if self.is_train else 'test'}"
#         image_base_dir = os.path.join(dataset_folder_path, image_folder_name)
#         if not os.path.isdir(image_base_dir): raise FileNotFoundError(f"未能找到预期的图像目录 {image_base_dir}")
#         logger.info(f"确认图像基目录: {image_base_dir}")

#         for idx, row in data_df.iterrows():
#             try: label = int(row[label_col]); assert label in [0, 1]
#             except: skipped_num += 1; continue
#             image_id = str(row[image_id_col]); content = str(row[content_col])
#             potential_image_paths = [ os.path.join(image_base_dir, f"{image_id}{ext}") for ext in ['', '.jpg', '.png', '.jpeg']]
#             full_image_path = next((p for p in potential_image_paths if os.path.exists(p) and os.path.isfile(p)), None)
#             if full_image_path is None: skipped_num += 1; continue
#             record = {"image_path": full_image_path, "label": label, "content": content, "category": 0}
#             self.label_dict.append(record)
#             if record["label"] == 0 and self.is_train and self.duplicate_fake_times > 0: # 重复 Fake (标签=0)
#                 for _ in range(self.duplicate_fake_times): self.label_dict.append(record)
#         if skipped_num > 0: logger.warning(f"因图像文件不存在或标签无效而跳过的样本总数: {skipped_num}")
#         logger.info(f"最终数据集大小 (包括重复): {len(self.label_dict)}")
#         if not self.label_dict: raise ValueError("数据集为空。")

#         # 初始化图像转换 (主要为 MAE)
#         self.mae_transform = transforms.Compose([
#             transforms.Resize((self.image_size, self.image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#     def __getitem__(self, index):
#         record = self.label_dict[index]
#         image_path = record["image_path"]
#         content = record["content"]
#         label = record["label"]
#         category = record["category"]

#         try: img_pil = Image.open(image_path).convert('RGB')
#         except Exception as e: logger.warning(f"无法加载图像 {image_path}: {e}，使用默认黑色图像。"); img_pil = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

#         try: img_mae = self.mae_transform(img_pil)
#         except Exception as e: logger.warning(f"MAE 图像变换失败 for {image_path}: {e}, 使用零值"); img_mae = torch.zeros((3, self.image_size, self.image_size))

#         clip_pixel_values = None; clip_default_size = 224
#         if self.clip_processor and hasattr(self.clip_processor, 'image_processor'):
#             try:
#                 clip_h = self.clip_processor.image_processor.size.get('height', clip_default_size)
#                 clip_w = self.clip_processor.image_processor.size.get('width', clip_default_size)
#                 clip_processed = self.clip_processor(images=img_pil, return_tensors="pt")
#                 clip_pixel_values = clip_processed['pixel_values'].squeeze(0)
#             except Exception as e: logger.warning(f"CLIP 图像处理失败 for {image_path}: {e}, 使用零值"); clip_h = self.clip_processor.image_processor.size.get('height', clip_default_size); clip_w = self.clip_processor.image_processor.size.get('width', clip_default_size); clip_pixel_values = torch.zeros((3, clip_h, clip_w))
#         else: clip_pixel_values = torch.zeros((3, clip_default_size, clip_default_size))

#         bert_input_ids, bert_attention_mask = None, None
#         if self.bert_tokenizer:
#             try:
#                 bert_tokenized = self.bert_tokenizer(content, padding='max_length', truncation=True, max_length=self.bert_max_len, return_tensors="pt")
#                 bert_input_ids = bert_tokenized['input_ids'].squeeze(0)
#                 bert_attention_mask = bert_tokenized['attention_mask'].squeeze(0)
#             except Exception as e: logger.warning(f"BERT 文本处理失败: {e}，使用零值")
#         if bert_input_ids is None: bert_input_ids = torch.zeros(self.bert_max_len, dtype=torch.long); bert_attention_mask = torch.zeros(self.bert_max_len, dtype=torch.long)

#         clip_input_ids, clip_attention_mask = None, None
#         if self.clip_processor:
#             try:
#                 clip_text_processed = self.clip_processor(text=content, return_tensors="pt", padding='max_length', truncation=True, max_length=self.clip_max_len)
#                 clip_input_ids = clip_text_processed['input_ids'].squeeze(0)
#                 clip_attention_mask = clip_text_processed['attention_mask'].squeeze(0)
#             except Exception as e: logger.warning(f"CLIP 文本处理失败: {e}，使用零值")
#         if clip_input_ids is None: clip_input_ids = torch.zeros(self.clip_max_len, dtype=torch.long); clip_attention_mask = torch.zeros(self.clip_max_len, dtype=torch.long)

#         item = {'content': bert_input_ids,'content_masks': bert_attention_mask,'image': img_mae,'clip_image': clip_pixel_values,'clip_text': clip_input_ids,'clip_attention_mask': clip_attention_mask,'label': torch.tensor(label, dtype=torch.float),'category': torch.tensor(category, dtype=torch.long)}
#         return item

#     def __len__(self):
#         return len(self.label_dict)


import random
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import logging
from transformers import BertTokenizer, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class FakeNet_dataset(data.Dataset):
    def __init__(
        self,
        root_path,
        bert_tokenizer_instance,
        clip_processor_instance,
        dataset_name="gossip",
        image_size=224,
        is_train=True,
        data_augment=False,
        duplicate_fake_times=0,
        bert_max_len=197,
        clip_max_len=77,
        reasoning_csv_path=None
    ):
        self.bert_tokenizer = bert_tokenizer_instance
        self.clip_processor = clip_processor_instance
        if self.bert_tokenizer is None:
            logger.warning("传入的 BERT tokenizer 实例为 None。")
        if self.clip_processor is None:
            logger.warning("传入的 CLIP processor 实例为 None。")
        
        self.duplicate_fake_times = duplicate_fake_times
        self.dataset_name = dataset_name.lower()
        assert (
            self.dataset_name == "politi" or self.dataset_name == "gossip"
        ), "错误! 只支持 'gossip' 或 'politi'!"
        
        super(FakeNet_dataset, self).__init__()
        logger.info(f"FakeNet_dataset 初始化: dataset_name='{self.dataset_name}', is_train={is_train}")
        self.is_train = is_train
        self.root_path = root_path
        self.image_size = image_size
        self.bert_max_len = bert_max_len
        self.clip_max_len = clip_max_len
        self.label_dict = []
        self.reasoning_df = None  # 推理嵌入数据框
        self.embedding_dim = 320  # 与模型输出特征维度保持一致
        
        # 1. 加载主数据集 CSV
        dataset_folder_path = self.root_path
        csv_file_name = f"gossip_{'train' if self.is_train else 'test'}.csv"
        csv_path = os.path.join(dataset_folder_path, csv_file_name)
        logger.info(f"从 CSV 加载数据: {csv_path}")
        try:
            data_df = pd.read_csv(csv_path)
            data_df = data_df.fillna('')  # 填充空值，避免后续处理报错
        except Exception as e:
            logger.error(f"读取 CSV 文件 {csv_path} 失败: {e}")
            raise
        
        # 2. 加载推理嵌入 CSV（关键修改：去重+确保单值）
        if reasoning_csv_path and os.path.exists(reasoning_csv_path):
            logger.info(f"从以下路径加载 reasoning embeddings: {reasoning_csv_path}")
            try:
                temp_df = pd.read_csv(reasoning_csv_path)
                temp_df = temp_df.fillna('')  # 填充空值
                
                # 关键步骤1：去除重复的image_id（保留第一行）
                if temp_df.duplicated(subset=['image_id']).any():
                    duplicate_count = temp_df.duplicated(subset=['image_id']).sum()
                    logger.warning(f"Reasoning CSV 存在 {duplicate_count} 个重复 image_id，已保留第一行")
                    temp_df = temp_df.drop_duplicates(subset=['image_id'], keep='first')
                
                # 关键步骤2：用image_id作为索引（此时每个image_id对应唯一行）
                self.reasoning_df = temp_df.set_index('image_id')
                logger.info(f"Reasoning CSV 已加载（去重后共 {len(self.reasoning_df)} 条数据），使用 'image_id' 作为索引。")
            except Exception as e:
                logger.error(f"读取 reasoning CSV 文件 {reasoning_csv_path} 失败: {e}")
                raise
        elif reasoning_csv_path:
            logger.warning(f"提供了 reasoning CSV 路径，但文件未找到: {reasoning_csv_path}")
        
        # 3. 构建标签字典（匹配图像路径、标签、内容）
        skipped_num = 0
        image_folder_name = f"{self.dataset_name}_{'train' if self.is_train else 'test'}"
        image_base_dir = os.path.join(dataset_folder_path, image_folder_name)
        
        for idx, row in data_df.iterrows():
            try:
                # 验证标签有效性（必须是 0 或 1）
                label = int(row['label'])
                assert label in [0, 1], f"无效标签: {label}，必须是 0 或 1"
                
                # 提取图像ID和文本内容（处理可能的列名差异）
                image_id = str(row['image_id'] if 'image_id' in row else row['image'])
                content = str(row['post_text'] if 'post_text' in row else row['content'])
                
                # 跳过空图像ID
                if not image_id.strip():
                    skipped_num += 1
                    continue
                
                # 匹配图像文件（兼容多种后缀）
                potential_extensions = ['', '.jpg', '.png', '.jpeg', '.gif']
                potential_image_paths = [
                    os.path.join(image_base_dir, f"{image_id}{ext}") 
                    for ext in potential_extensions
                ]
                # 找到第一个存在的图像文件
                full_image_path = None
                for path in potential_image_paths:
                    if os.path.exists(path) and os.path.isfile(path):
                        full_image_path = path
                        break
                
                # 跳过无图像文件的样本
                if full_image_path is None:
                    logger.debug(f"未找到 image_id {image_id} 的图像文件，跳过")
                    skipped_num += 1
                    continue
                
                # 添加到标签字典
                record = {
                    "id": image_id,
                    "image_path": full_image_path,
                    "label": label,
                    "content": content,
                    "category": 0  # GossipCop 数据集固定类别为 0
                }
                self.label_dict.append(record)
                
                # 训练时重复伪造样本（可选增强）
                if record["label"] == 0 and self.is_train and self.duplicate_fake_times > 0:
                    for _ in range(self.duplicate_fake_times):
                        self.label_dict.append(record.copy())
            
            except Exception as e:
                # 捕获所有异常，避免单一样本错误中断整个加载流程
                logger.debug(f"处理行 {idx} 时出错: {e}，跳过该样本")
                skipped_num += 1
                continue
        
        # 日志输出加载统计
        logger.info(f"最终数据集大小 (包括重复样本): {len(self.label_dict)}")
        if skipped_num > 0:
            logger.warning(f"因无效数据跳过的样本总数: {skipped_num}")
        
        # 4. 定义 MAE 图像预处理管道
        self.mae_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _parse_reasoning_embedding(self, emb_str):
        """
        私有方法：解析推理嵌入字符串（处理空格分隔、方括号、异常值）
        :param emb_str: 嵌入字符串（格式如 "[-3.0986e-01  3.1996e-01 ...]"）
        :return: 320维 torch.Tensor（异常时返回零向量）
        """
        # 1. 处理非字符串类型（如 Series、数字）
        if not isinstance(emb_str, str):
            # 若为Series（理论上去重后不会出现），取第一个值并转字符串
            if hasattr(emb_str, 'iloc'):
                emb_str = str(emb_str.iloc[0]) if len(emb_str) > 0 else ''
            else:
                emb_str = str(emb_str)  # 强制转为字符串
        emb_str = emb_str.strip()
        
        # 2. 处理空值或无效字符串（包括包含"image_id"的错误格式）
        invalid_markers = ['', 'nan', 'None', '[]', 'image_id']
        if any(marker in emb_str for marker in invalid_markers):
            logger.debug(f"无效嵌入字符串: '{emb_str[:50]}...'，返回零向量")
            return torch.zeros(self.embedding_dim, dtype=torch.float)
        
        # 3. 去除方括号（核心：处理 "[x y z]" 格式）
        if emb_str.startswith('['):
            emb_str = emb_str[1:]  # 去掉左括号
        if emb_str.endswith(']'):
            emb_str = emb_str[:-1]  # 去掉右括号
        emb_str = emb_str.strip()
        
        # 4. 按空格分割（兼容多个连续空格），过滤非数字字符
        emb_parts = [part.strip() for part in emb_str.split() if part.strip().replace('.', '').replace('-', '').isdigit()]
        
        # 5. 转换为浮点数并对齐维度
        try:
            emb_float = [float(part) for part in emb_parts]
            # 维度不足补零，超出截断
            if len(emb_float) < self.embedding_dim:
                emb_float += [0.0] * (self.embedding_dim - len(emb_float))
            elif len(emb_float) > self.embedding_dim:
                emb_float = emb_float[:self.embedding_dim]
            return torch.tensor(emb_float, dtype=torch.float)
        except ValueError as e:
            logger.error(f"解析嵌入字符串失败: '{emb_str[:50]}...'，错误: {e}，返回零向量")
            return torch.zeros(self.embedding_dim, dtype=torch.float)
    
    def __getitem__(self, index):
        """
        获取单个样本：处理图像、文本编码、推理嵌入
        :param index: 样本索引
        :return: 包含所有特征的字典（异常时返回 None）
        """
        try:
            record = self.label_dict[index]
            image_path = record["image_path"]
            content = record["content"]
            label = record["label"]
            category = record["category"]
            image_id = record["id"]
            
            # 1. 处理图像（MAE 输入）
            try:
                img_pil = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.warning(f"打开图像 {image_path} 失败: {e}，使用空白图像替代")
                img_pil = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
            img_mae = self.mae_transform(img_pil)
            
            # 2. 处理文本编码（BERT + CLIP）
            try:
                clip_processed = self.clip_processor(
                    images=img_pil,
                    text=content,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=self.clip_max_len
                )
                bert_tokenized = self.bert_tokenizer(
                    content,
                    padding='max_length',
                    truncation=True,
                    max_length=self.bert_max_len,
                    return_tensors="pt"
                )
            except Exception as e:
                logger.error(f"处理 image_id {image_id} 的编码时出错: {e}")
                return None
            
            # 3. 处理推理嵌入（核心：去重后查询+鲁棒解析）
            text_reasoning_emb = torch.zeros(self.embedding_dim, dtype=torch.float)
            image_reasoning_emb = torch.zeros(self.embedding_dim, dtype=torch.float)
            cross_modal_reasoning_emb = torch.zeros(self.embedding_dim, dtype=torch.float)
            
            if self.reasoning_df is not None:
                try:
                    # 按image_id查询（去重后确保返回单行数据）
                    reasoning_data = self.reasoning_df.loc[image_id]
                    
                    # 解析三个嵌入字段（调用鲁棒解析方法）
                    text_reasoning_emb = self._parse_reasoning_embedding(reasoning_data['text_reasoning_embedding'])
                    image_reasoning_emb = self._parse_reasoning_embedding(reasoning_data['image_reasoning_embedding'])
                    cross_modal_reasoning_emb = self._parse_reasoning_embedding(reasoning_data['cross_modal_reasoning_embedding'])
                    
                except KeyError:
                    logger.debug(f"image_id {image_id} 在 reasoning CSV 中未找到，使用零向量")
                except Exception as e:
                    logger.error(f"处理 image_id {image_id} 的推理嵌入时出错: {e}，使用零向量")
            
            # 4. 组装返回字典（挤压batch维度）
            return {
                'content': bert_tokenized['input_ids'].squeeze(0),
                'content_masks': bert_tokenized['attention_mask'].squeeze(0),
                'image': img_mae,
                'clip_image': clip_processed['pixel_values'].squeeze(0),
                'clip_text': clip_processed['input_ids'].squeeze(0),
                'clip_attention_mask': clip_processed['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.float),
                'category': torch.tensor(category, dtype=torch.long),
                'text_reasoning_embedding': text_reasoning_emb,
                'image_reasoning_embedding': image_reasoning_emb,
                'cross_modal_reasoning_embedding': cross_modal_reasoning_emb
            }
        
        except Exception as e:
            logger.error(f"获取样本 index {index} 时出错: {e}")
            return None
    
    def __len__(self):
        """返回数据集总长度"""
        return len(self.label_dict)