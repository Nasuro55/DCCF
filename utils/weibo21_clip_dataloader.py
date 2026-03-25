# # utils/weibo21_clip_dataloader.py

# import pickle
# import os
# import pandas as pd
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from transformers import BertTokenizer # 用于 BERT 文本
# import cn_clip.clip as cn_clip_module # 重命名以避免与变量名冲突
# from cn_clip.clip import load_from_name as cn_clip_load_from_name
# import logging
# import numpy as np # _init_fn 需要

# # 设置日志记录器
# logger = logging.getLogger(__name__)
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)


# def _init_fn(worker_id):
#     np.random.seed(2024 + worker_id) # 每个 worker 不同种子

# def word2input_updated(texts, tokenizer: BertTokenizer, max_len: int):
#     """
#     使用传入的 BertTokenizer 实例进行分词。
#     """
#     token_ids_list = []
#     attention_masks_list = []
#     for text in texts:
#         text_str = str(text) if text is not None else ""
#         encoded_dict = tokenizer.encode_plus(
#             text_str,
#             max_length=max_len,
#             add_special_tokens=True,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt' # 直接返回 PyTorch 张量
#         )
#         token_ids_list.append(encoded_dict['input_ids'].squeeze(0))
#         attention_masks_list.append(encoded_dict['attention_mask'].squeeze(0))

#     if not token_ids_list:
#         return torch.empty((0, max_len), dtype=torch.long), torch.empty((0, max_len), dtype=torch.long)

#     token_ids_tensor = torch.stack(token_ids_list)
#     masks_tensor = torch.stack(attention_masks_list)
#     return token_ids_tensor, masks_tensor


# class bert_data: # 这个类在 run.py 中被别名为 Weibo21DataLoaderClass
#     def __init__(self,
#                  max_len: int,
#                  batch_size: int,
#                  vocab_file: str, # BERT vocab.txt 的 *目录* 或 HuggingFace 模型名
#                  category_dict: dict,
#                  num_workers: int = 2,
#                  # 为 cn_clip 模型添加一个参数，如果不想硬编码 "ViT-B-16"
#                  cn_clip_model_name: str = "ViT-B-16",
#                  cn_clip_download_root: str = './'): # cn_clip 模型下载/缓存根目录

#         self.max_len = max_len
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.vocab_file_path = vocab_file # 保存路径以供参考
#         self.category_dict = category_dict
#         self.cn_clip_model = None
#         # self.cn_clip_preprocess = None # 预处理函数，如果图片也实时处理的话

#         logger.info(f"Weibo21 DataLoader (bert_data) 初始化:")
#         logger.info(f"  max_len: {self.max_len}")
#         logger.info(f"  batch_size: {self.batch_size}")
#         logger.info(f"  vocab_file (for BERT): {self.vocab_file_path}")
#         logger.info(f"  category_dict: {self.category_dict}")
#         logger.info(f"  num_workers: {self.num_workers}")
#         logger.info(f"  cn_clip_model_name: {cn_clip_model_name}")

#         try:
#             # 更新 BertTokenizer 加载方式
#             # vocab_file 参数现在应该是包含 vocab.txt 的目录路径，或者是 HuggingFace 模型名
#             # 从 main.py 来看，args.bert_vocab_file 指向 vocab.txt 文件本身。
#             # 我们需要它的父目录给 from_pretrained。
#             bert_tokenizer_path = os.path.dirname(self.vocab_file_path) if os.path.isfile(self.vocab_file_path) else self.vocab_file_path
#             if not os.path.isdir(bert_tokenizer_path): # 如果它不是一个有效的目录（比如只是文件名但目录不存在）
#                  logger.warning(f"BERT tokenizer 路径 '{bert_tokenizer_path}' 不是一个有效目录。尝试直接使用 '{self.vocab_file_path}' 作为标识符。")
#                  bert_tokenizer_path = self.vocab_file_path # 回退到使用原始路径/名称

#             self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
#             logger.info(f"BERT Tokenizer 成功加载: {bert_tokenizer_path}")
#         except Exception as e:
#             logger.error(f"从 {bert_tokenizer_path} (源自 {self.vocab_file_path}) 加载 BERT Tokenizer 失败: {e}")
#             raise

#         # 在 __init__ 中加载 cn_clip 模型以提高效率
#         try:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             logger.info(f"尝试使用设备 '{device}' 加载 cn_clip 模型 '{cn_clip_model_name}'...")
#             # 注意：cn_clip_load_from_name 可能需要网络连接来下载模型（如果本地没有）
#             # 如果你的服务器无法访问 huggingface.co，这里可能会失败
#             self.cn_clip_model, _ = cn_clip_load_from_name(
#                 cn_clip_model_name,
#                 device=device,
#                 download_root=cn_clip_download_root
#             )
#             # self.cn_clip_preprocess = preprocess # 如果需要实时处理CLIP图片
#             logger.info(f"cn_clip 模型 '{cn_clip_model_name}' 加载成功。")
#         except Exception as e:
#             self.cn_clip_model = None # 加载失败则设为None
#             # self.cn_clip_preprocess = None
#             logger.warning(f"加载 cn_clip 模型 '{cn_clip_model_name}' 失败: {e}. "
#                            "CLIP 文本分词将不可用，除非 cn_clip.tokenize 能在无模型实例情况下工作（不太可能）。")
#             # 如果CLIP文本是必须的，这里应该 raise error。
#             # 根据用户代码，clip.tokenize() 是一个静态方法或全局方法，可能不依赖self.cn_clip_model实例。
#             # cn_clip.tokenize 是 cn_clip.clip.tokenize，它通常需要知道词汇表和上下文长度，
#             # 这些通常与加载的模型相关联。如果它是真正的静态方法，则可能使用默认设置。
#             # 查阅 cn_clip 文档，cn_clip.tokenize() 是一个函数，可以直接调用。
#             # 它不需要模型实例，但可能需要预先下载的词汇表。

#     def load_data(self,
#                   excel_path: str,          # .xlsx 文件路径
#                   mae_image_pkl_path: str,  # MAE 特征 PKL 文件路径
#                   clip_image_pkl_path: str, # CLIP 特征 PKL 文件路径
#                   shuffle: bool):
#         """
#         加载数据并返回 DataLoader。
#         现在假设 mae_image_pkl_path 和 clip_image_pkl_path 中的 PKL 文件包含单个堆叠的张量。
#         """
#         logger.info(f"开始加载数据: excel_path='{excel_path}'")
#         logger.info(f"  MAE PKL: '{mae_image_pkl_path}', CLIP PKL: '{clip_image_pkl_path}'")

#         try:
#             data_df = pd.read_excel(excel_path)
#             # 假设文本列为 'content', 标签列为 'label', 类别列为 'category'
#             # !! 请根据你的 Weibo21 Excel 文件调整这些列名 !!
#             text_col = 'content'
#             label_col = 'label'
#             category_col = 'category'

#             if text_col not in data_df.columns:
#                 logger.error(f"Excel 文件 {excel_path} 中缺少文本列 '{text_col}'。")
#                 return None
#             if label_col not in data_df.columns:
#                 logger.error(f"Excel 文件 {excel_path} 中缺少标签列 '{label_col}'。")
#                 return None
#             if category_col not in data_df.columns: # 类别列是可选的，如果没有，则使用默认
#                 logger.warning(f"Excel 文件 {excel_path} 中缺少类别列 '{category_col}'。将使用默认分类。")


#             data_df[text_col] = data_df[text_col].fillna('') # 填充 NaN 文本
#             logger.info(f"从 {excel_path} 加载了 {len(data_df)} 行。")
#         except FileNotFoundError:
#             logger.error(f"Excel 文件未找到: {excel_path}")
#             return None
#         except Exception as e:
#             logger.error(f"读取 Excel 文件 {excel_path} 失败: {e}")
#             return None

#         # 1. 处理 BERT 文本
#         content_texts = data_df[text_col].tolist()
#         bert_token_ids, bert_masks = word2input_updated(content_texts, self.bert_tokenizer, self.max_len)
#         logger.info(f"BERT 文本处理完成。Shape: {bert_token_ids.shape}")

#         # 2. 处理标签
#         try:
#             labels_tensor = torch.tensor(data_df[label_col].astype(int).to_numpy(), dtype=torch.long)
#             logger.info(f"标签处理完成。Shape: {labels_tensor.shape}")
#         except Exception as e:
#             logger.error(f"处理标签失败: {e}")
#             return None

#         # 3. 处理类别
#         if category_col in data_df.columns:
#             try:
#                 categories_tensor = torch.tensor(
#                     data_df[category_col].astype(str).apply(lambda c: self.category_dict.get(c, 0)).to_numpy(), # 使用 get 获取，提供默认值0
#                     dtype=torch.long
#                 )
#                 logger.info(f"类别处理完成。Shape: {categories_tensor.shape}")
#             except Exception as e:
#                 logger.error(f"处理类别失败: {e}. 将使用默认类别0。")
#                 categories_tensor = torch.zeros(len(data_df), dtype=torch.long)
#         else:
#             logger.info("未找到类别列，使用默认类别0。")
#             categories_tensor = torch.zeros(len(data_df), dtype=torch.long)


#         # 4. 加载 MAE 图像特征 (预处理好的单个堆叠张量)
#         try:
#             with open(mae_image_pkl_path, 'rb') as f:
#                 mae_image_tensor = pickle.load(f)
#             if not isinstance(mae_image_tensor, torch.Tensor):
#                 logger.error(f"MAE PKL 文件 ({mae_image_pkl_path}) 未包含有效的 PyTorch 张量。类型: {type(mae_image_tensor)}")
#                 return None
#             logger.info(f"MAE 图像特征 PKL 加载成功。Shape: {mae_image_tensor.shape}")
#         except FileNotFoundError:
#             logger.error(f"MAE 图像 PKL 文件未找到: {mae_image_pkl_path}")
#             return None
#         except Exception as e:
#             logger.error(f"加载 MAE 图像 PKL ({mae_image_pkl_path}) 失败: {e}")
#             return None

#         # 5. 加载 CLIP 图像特征 (预处理好的单个堆叠张量)
#         try:
#             with open(clip_image_pkl_path, 'rb') as f:
#                 clip_image_tensor = pickle.load(f)
#             if not isinstance(clip_image_tensor, torch.Tensor):
#                 logger.error(f"CLIP PKL 文件 ({clip_image_pkl_path}) 未包含有效的 PyTorch 张量。类型: {type(clip_image_tensor)}")
#                 return None
#             logger.info(f"CLIP 图像特征 PKL 加载成功。Shape: {clip_image_tensor.shape}")
#         except FileNotFoundError:
#             logger.error(f"CLIP 图像 PKL 文件未找到: {clip_image_pkl_path}")
#             return None
#         except Exception as e:
#             logger.error(f"加载 CLIP 图像 PKL ({clip_image_pkl_path}) 失败: {e}")
#             return None

#         # 6. 处理 CLIP 文本 (使用 cn_clip.tokenize)
#         # cn_clip.tokenize 通常返回一个 tensor [N, context_length]
#         # 注意：cn_clip.tokenize 可能有其自己的最大长度限制 (通常是76或77)
#         # 如果你的 self.max_len 用于 BERT，CLIP 的文本长度可能不同。
#         # 这里的 content_texts 是原始文本列表
#         try:
#             # 对于 cn_clip.tokenize, 如果文本数量很多，一次性处理可能消耗大量内存或非常慢
#             # 如果遇到性能问题，可以考虑分批处理 content_texts 然后 torch.cat
#             # 另外，确保 cn_clip 使用的词汇表与你的中文内容兼容
#             logger.info(f"开始使用 cn_clip.tokenize 处理 {len(content_texts)} 条 CLIP 文本...")
#             clip_text_tensor = cn_clip_module.tokenize(content_texts, context_length=77) # 明确指定 context_length
#             if not isinstance(clip_text_tensor, torch.Tensor): # tokenize 可能返回numpy array
#                 clip_text_tensor = torch.tensor(clip_text_tensor, dtype=torch.long)
#             logger.info(f"CLIP 文本处理完成。Shape: {clip_text_tensor.shape}")
#         except Exception as e:
#             logger.error(f"使用 cn_clip.tokenize 处理文本失败: {e}")
#             logger.error("确保 cn_clip 库已正确安装并且其依赖项（如词汇表）可用。")
#             # 可以考虑返回占位符，或者直接失败
#             # clip_text_tensor = torch.zeros((len(content_texts), 77), dtype=torch.long) # 假设context_length是77
#             return None


#         # 7. 验证所有张量的第一个维度（样本数）是否一致
#         num_samples = len(data_df)
#         if not (bert_token_ids.shape[0] == num_samples and
#                 bert_masks.shape[0] == num_samples and
#                 labels_tensor.shape[0] == num_samples and
#                 categories_tensor.shape[0] == num_samples and
#                 mae_image_tensor.shape[0] == num_samples and
#                 clip_image_tensor.shape[0] == num_samples and
#                 clip_text_tensor.shape[0] == num_samples):
#             logger.error("一个或多个处理后的数据张量样本数量与源数据不匹配！")
#             logger.error(f"  Excel行数: {num_samples}")
#             logger.error(f"  BERT IDs: {bert_token_ids.shape[0]}, BERT Masks: {bert_masks.shape[0]}")
#             logger.error(f"  Labels: {labels_tensor.shape[0]}, Categories: {categories_tensor.shape[0]}")
#             logger.error(f"  MAE Images: {mae_image_tensor.shape[0]}, CLIP Images: {clip_image_tensor.shape[0]}")
#             logger.error(f"  CLIP Text: {clip_text_tensor.shape[0]}")
#             return None

#         # 8. 创建 TensorDataset
#         # 顺序必须与 utils.py 中的 clipdata2gpu 函数期望的一致
#         # 'content': batch[0] -> bert_token_ids
#         # 'content_masks': batch[1] -> bert_masks
#         # 'label': batch[2] -> labels_tensor
#         # 'category': batch[3] -> categories_tensor
#         # 'image':batch[4] -> mae_image_tensor (MAE 图像)
#         # 'clip_image':batch[5] -> clip_image_tensor (CLIP 图像)
#         # 'clip_text': batch[6] -> clip_text_tensor (CLIP 文本)
#         # 你的 utils.py 中的 clipdata2gpu 期望 7 个元素。
#         # 如果 TensorDataset 返回的元组少于7个元素，那里会报错。
#         # 确保这里有7个对应的张量。
#         # 如果 utils.py 中的 clipdata2gpu 需要 clip_attention_mask，这里也需要加上。
#         # cn_clip.tokenize 通常不直接返回 attention_mask, 它返回的是 padded input_ids.
#         # CLIP模型的forward pass会自行处理padding。
#         # 为了与你 utils.py 中的 clipdata2gpu 期望的第8个元素 (clip_attention_mask) 匹配，
#         # 如果 cn_clip.tokenize 不返回mask，我们需要创建一个伪mask或者调整clipdata2gpu。
#         # 对于CLIP，通常 input_ids 中 padding token 为0，非padding为其他值，
#         # 可以据此生成attention_mask: (clip_text_tensor != 0).long()
#         clip_attention_mask_tensor = (clip_text_tensor != 0).long() # 生成注意力掩码 (0 for padding, 1 for non-padding)

#         datasets = TensorDataset(
#             bert_token_ids,
#             bert_masks,
#             labels_tensor,
#             categories_tensor,
#             mae_image_tensor,
#             clip_image_tensor,
#             clip_text_tensor,
#             clip_attention_mask_tensor # 添加 CLIP attention mask
#         )
#         logger.info(f"TensorDataset 创建成功，包含 {len(datasets)} 个样本。")

#         # 9. 创建 DataLoader
#         dataloader = DataLoader(
#             dataset=datasets,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             shuffle=shuffle,
#             worker_init_fn=_init_fn
#         )
#         logger.info(f"DataLoader 创建成功，批大小 {self.batch_size}，共 {len(dataloader)} 个批次。")
#         return dataloader

# -*-codeing = utf-8 -*-

import pickle
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image

def read_image():
    image_list = {}
    file_list = ['data/nonrumor_images/', 'data/rumor_images/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                #im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print("wrong"+filename)
    print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list

def _init_fn(worker_id):
    np.random.seed(2024)

def read_pkl(path):
    with open(path,"rb")as f:
        t = pickle.load(f)
    return t
def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

def word2input(texts,vocab_file,max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids =[]
    for i,text in enumerate(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i,token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids,masks

class bert_data():
    def __init__(self,max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self,path,imagepath,clipimagepath,shuffle,text_only = False):
        self.data = pd.read_excel(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clipmodel, _ = load_from_name("ViT-B-16", device=device, download_root='./')
        content = self.data['content'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(self.data['category'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())
        token_ids, masks = word2input(content,self.vocab_file,self.max_len)
        ordered_image = pickle.load(open(imagepath,'rb'))
        clip_image = pickle.load(open(clipimagepath, 'rb'))
        clip_text = clip.tokenize(content)
        #("token_ids",token_ids.size())
        #print("masks", masks.size())
        #print("label", label.size())
        #print("category", category.size())
        #print("ordered_image", ordered_image.size())
        #print("clip_image", clip_image.size())
        #print("clip_text", clip_text.size())
        datasets =TensorDataset(token_ids,
                                masks,
                                label,
                                category,
                                ordered_image,
                                clip_image,
                                clip_text
        )
        dataloader = DataLoader(
            dataset = datasets,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = shuffle,
            worker_init_fn = _init_fn
        )
        return dataloader
