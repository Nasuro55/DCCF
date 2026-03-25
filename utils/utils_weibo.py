# # -*-codeing = utf-8 -*-

# # -*-codeing = utf-8 -*-

import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np

# --- START: MODIFIED CODE ---
# 修改了 clipdata2gpu 和 data2gpu 函数，使其能够接收 use_cuda 参数

def clipdata2gpu(batch, use_cuda):
    """
    根据 use_cuda 标志，有条件地将数据移动到 GPU。
    """
    batch_data = {
        'content': batch[0],
        'content_masks': batch[1],
        'label': batch[2],
        'category': batch[3],
        'image': batch[4],
        'clip_image': batch[5],
        'clip_text': batch[6]
    }
    if use_cuda:
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.cuda()
    
    # 关键修改：为蒸馏任务添加 teacher reasoning embeddings 的处理
    # 假设它们在 batch 的第7、8、9个位置
    if len(batch) > 7 and batch[7] is not None:
        batch_data['teacher_reasoning_text_emb'] = batch[7].cuda() if use_cuda else batch[7]
    if len(batch) > 8 and batch[8] is not None:
        batch_data['teacher_reasoning_image_emb'] = batch[8].cuda() if use_cuda else batch[8]
    if len(batch) > 9 and batch[9] is not None:
        batch_data['teacher_reasoning_cross_emb'] = batch[9].cuda() if use_cuda else batch[9]

    return batch_data

def data2gpu(batch, use_cuda):
    """
    根据 use_cuda 标志，有条件地将数据移动到 GPU。
    """
    batch_data = {
        'content': batch[0],
        'content_masks': batch[1],
        'label': batch[2],
        'category': batch[3],
        'image': batch[4]
    }
    if use_cuda:
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.cuda()
    return batch_data

# --- END: MODIFIED CODE ---


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    # 处理 category 可能为空列表的情况
    if category:
        for i, c in enumerate(category):
            c_val = c.item() if hasattr(c, 'item') else c
            category_name = reverse_category_dict[c_val]
            res_by_category[category_name]['y_true'].append(y_true[i])
            res_by_category[category_name]['y_pred'].append(y_pred[i])

    # 计算整体指标
    try:
        metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        metrics_by_category['auc'] = 0.0

    y_pred_int = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['metric'] = f1_score(y_true, y_pred_int, average='macro')
    metrics_by_category['recall'] = recall_score(y_true, y_pred_int, average='macro')
    metrics_by_category['precision'] = precision_score(y_true, y_pred_int, average='macro', zero_division=0)
    metrics_by_category['acc'] = accuracy_score(y_true, y_pred_int)
    metrics_by_category['f1'] = metrics_by_category['metric'] # 别名 F1

    # 计算各类别指标
    for c, res in res_by_category.items():
        if not res['y_true']:
            continue
        
        y_pred_cat_int = np.around(np.array(res['y_pred'])).astype(int)
        
        cat_auc = 0.0
        try:
            # AUC 需要至少两个类别
            if len(np.unique(res['y_true'])) > 1:
                cat_auc = roc_auc_score(res['y_true'], res['y_pred'])
        except ValueError:
            pass

        metrics_by_category[c] = {
            'precision': precision_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
            'recall': recall_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
            'fscore': f1_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
            'auc': cat_auc,
            'acc': accuracy_score(res['y_true'], y_pred_cat_int)
        }
    return metrics_by_category


def metricsTrueFalse(y_true, y_pred, category, category_dict):
    """
    此函数现在直接调用修正后的 metrics 函数，以提供统一和健壮的格式。
    """
    results = metrics(y_true, y_pred, category, category_dict)

    # 分别计算真新闻和假新闻的指标
    y_true_np = np.array(y_true)
    y_pred_int = np.around(np.array(y_pred)).astype(int)

    # 真实新闻 (label=0)
    real_indices = np.where(y_true_np == 0)[0]
    if len(real_indices) > 0:
        real_true = y_true_np[real_indices]
        real_pred = y_pred_int[real_indices]
        results['Real_Acc'] = accuracy_score(real_true, real_pred)
        results['Real_Pre'] = precision_score(real_true, real_pred, pos_label=0, zero_division=0)
        results['Real_Rec'] = recall_score(real_true, real_pred, pos_label=0, zero_division=0)
        results['Real_F1'] = f1_score(real_true, real_pred, pos_label=0, zero_division=0)

    # 虚假新闻 (label=1)
    fake_indices = np.where(y_true_np == 1)[0]
    if len(fake_indices) > 0:
        fake_true = y_true_np[fake_indices]
        fake_pred = y_pred_int[fake_indices]
        results['Fake_Acc'] = accuracy_score(fake_true, fake_pred)
        results['Fake_Pre'] = precision_score(fake_true, fake_pred, pos_label=1, zero_division=0)
        results['Fake_Rec'] = recall_score(fake_true, fake_pred, pos_label=1, zero_division=0)
        results['Fake_F1'] = f1_score(fake_true, fake_pred, pos_label=1, zero_division=0)

    # 宏平均指标
    results['Macro_Acc'] = results.get('acc', 0)
    results['Macro_Pre'] = results.get('precision', 0)
    results['Macro_Rec'] = results.get('recall', 0)
    results['Macro_F1'] = results.get('f1', 0)
    
    return results


class Recorder():

    def __init__(self, early_step, metric_key='metric'):
        self.max = {metric_key: 0}
        self.cur = {metric_key: 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step
        self.metric_key = metric_key

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("current", self.cur)
        return self.judge()

    def judge(self):
        # 确保比较的键存在
        if self.cur.get(self.metric_key, 0) > self.max.get(self.metric_key, 0):
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

# # -*-codeing = utf-8 -*-

# import torch
# from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
# import numpy as np

# ## Data Handling
# def data_to_gpu(data, use_cuda):
#     """
#     Moves a dictionary or list containing tensors to the GPU, if available.
#     """
#     if not use_cuda:
#         return data

#     if isinstance(data, dict):
#         for key, value in data.items():
#             if isinstance(value, torch.Tensor):
#                 data[key] = value.cuda()
#     elif isinstance(data, list):
#         data = [item.cuda() if isinstance(item, torch.Tensor) else item for item in data]
    
#     return data

# ## Metrics Calculation
# def metrics(y_true, y_pred, category, category_dict):
#     """
#     This helper function calculates various metrics and organizes them by category.
#     """
#     res_by_category = {}
#     metrics_by_category = {}
#     reverse_category_dict = {}
#     for k, v in category_dict.items():
#         reverse_category_dict[v] = k
#         res_by_category[k] = {"y_true": [], "y_pred": []}

#     # Populates the res_by_category dictionary with true and predicted values for each category
#     if category:
#         for i, c in enumerate(category):
#             c_val = c.item() if hasattr(c, 'item') else c
#             category_name = reverse_category_dict[c_val]
#             res_by_category[category_name]['y_true'].append(y_true[i])
#             res_by_category[category_name]['y_pred'].append(y_pred[i])

#     # Calculates overall metrics
#     try:
#         metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
#     except ValueError:
#         metrics_by_category['auc'] = 0.0
#         pass
    
#     y_pred_int = np.around(np.array(y_pred)).astype(int)
#     metrics_by_category['metric'] = f1_score(y_true, y_pred_int, average='macro')
#     metrics_by_category['recall'] = recall_score(y_true, y_pred_int, average='macro')
#     metrics_by_category['precision'] = precision_score(y_true, y_pred_int, average='macro', zero_division=0)
#     metrics_by_category['acc'] = accuracy_score(y_true, y_pred_int)
#     metrics_by_category['f1'] = metrics_by_category['metric']

#     # Calculates metrics by category
#     for c, res in res_by_category.items():
#         if not res['y_true']:
#             continue
        
#         y_pred_cat_int = np.around(np.array(res['y_pred'])).astype(int)
        
#         cat_auc = 0.0
#         try:
#             if len(np.unique(res['y_true'])) > 1:
#                 cat_auc = roc_auc_score(res['y_true'], res['y_pred'])
#         except ValueError:
#             pass

#         metrics_by_category[c] = {
#             'precision': precision_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
#             'recall': recall_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
#             'fscore': f1_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
#             'auc': cat_auc,
#             'acc': accuracy_score(res['y_true'], y_pred_cat_int)
#         }

#     return metrics_by_category


# def metricsTrueFalse(y_true, y_pred, category, category_dict):
#     """
#     This function calls the metrics function to get the output format.
#     """
#     return metrics(y_true, y_pred, category, category_dict)

# ## Training Utilities
# class Averager():
#     """
#     A class to calculate the running average of a value.
#     """
#     def __init__(self):
#         self.n = 0
#         self.v = 0

#     def add(self, x):
#         self.v = (self.v * self.n + x) / (self.n + 1)
#         self.n += 1

#     def item(self):
#         return self.v


# class Recorder():
#     """
#     A class to record and manage model training progress for early stopping.
#     """
#     def __init__(self, early_step):
#         self.max = {'metric': 0}
#         self.cur = {'metric': 0}
#         self.maxindex = 0
#         self.curindex = 0
#         self.early_step = early_step

#     def add(self, x):
#         self.cur = x
#         self.curindex += 1
#         print("curent", self.cur)
#         return self.judge()

#     def judge(self):
#         if self.cur['metric'] > self.max['metric']:
#             self.max = self.cur
#             self.maxindex = self.curindex
#             self.showfinal()
#             return 'save'
#         self.showfinal()
#         if self.curindex - self.maxindex >= self.early_step:
#             return 'esc'
#         else:
#             return 'continue'

#     def showfinal(self):
#         print("Max", self.max)