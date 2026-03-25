# python visualize_tsne.py --dataset weibo21 --batchsize 64

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import argparse
import logging

# --- 1. 项目代码导入 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    from model.domain_weibo21 import MultiDomainPLEFENDModel
    logger.info("成功从 'model.domain_weibo21' 导入 MultiDomainPLEFENDModel。")
except ImportError as e:
    logger.error(f"无法从 'model/domain_weibo21.py' 导入 MultiDomainPLEFENDModel！错误: {e}")
    exit()
try:
    from run import Run
    logger.info("成功从 'run.py' 导入 Run 类。")
except ImportError as e:
    logger.error(f"无法导入 Run 类！错误: {e}")
    exit()

# --- 2. 参数配置 ---
parser = argparse.ArgumentParser(description="为指定数据集的模型生成 t-SNE 可视化")
parser.add_argument('--dataset', default='weibo21', choices=['weibo', 'weibo21'], help="要可视化的数据集。")
parser.add_argument('--batchsize', type=int, default=64, help="数据加载时的批处理大小。")
parser.add_argument('--gpu', default='0', help="要使用的 GPU ID。")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {DEVICE}")

# --- 3. 加载项目配置并准备数据加载器 ---
logger.info(f"正在为数据集 '{args.dataset}' 生成配置...")
if args.dataset == 'weibo':
    config = {
        'dataset': 'weibo', 'model_name': 'domain_weibo', 'weibo_data_dir': './data/',
        'bert_model_path_weibo': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
        'bert_vocab_file_weibo': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
        'batchsize': args.batchsize, 'max_len': 197, 'num_workers': 4, 'emb_dim': 768,
        'lr': 0.000175, 'early_stop': 100, 'epoch': 50, 'save_param_dir': './param_model',
        'model_params': {'mlp': {'dims': [768, 256], 'dropout': 0.2}},
        'vocab_file': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
        'bert': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
    }
elif args.dataset == 'weibo21':
    config = {
        'dataset': 'weibo21', 'model_name': 'domain_weibo', 'weibo21_data_dir': './Weibo_21/',
        'bert_model_path_weibo': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
        'bert_vocab_file_weibo': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
        'batchsize': args.batchsize, 'max_len': 197, 'num_workers': 4, 'emb_dim': 768,
        'lr': 0.0005, 'early_stop': 100, 'epoch': 50, 'save_param_dir': './param_model',
        'model_params': {'mlp': {'dims': [384], 'dropout': 0.2}},
        'vocab_file': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
        'bert': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
    }
else:
    raise ValueError("此脚本目前仅支持 'weibo' 和 'weibo21' 数据集。")

# --- 4 & 5. 模型加载和特征提取 ---
logger.info("初始化 Run 类以获取数据加载器...")
try:
    runner = Run(config)
    _, _, test_loader = runner.get_dataloader()
    logger.info("成功获取 test_loader。")
except Exception as e:
    logger.error(f"在获取 dataloader 时出错！错误: {e}", exc_info=True)
    exit()

logger.info("正在初始化模型...")
final_mlp_dims = config['model_params']['mlp']['dims'] + [1]
logger.info(f"为 {args.dataset} 数据集构建的MLP维度为: {final_mlp_dims}")
model = MultiDomainPLEFENDModel(
    emb_dim=config['emb_dim'], mlp_dims=final_mlp_dims, bert=config['bert'],
    out_channels=320, dropout=config['model_params']['mlp']['dropout']
).to(DEVICE)

try:
    MODEL_WEIGHTS_PATH = os.path.join(config['save_param_dir'], f"{config['dataset']}_{config['model_name']}", 'parameter_clip111.pkl')
    logger.info(f"尝试从路径加载权重: {MODEL_WEIGHTS_PATH}")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"错误：找不到模型权重文件 '{MODEL_WEIGHTS_PATH}'。")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE), strict=False)
    logger.info("模型权重加载成功（非严格模式）！")
except Exception as e:
    logger.error(f"加载模型权重时发生未知错误: {e}")
    exit()

model.eval()
logger.info("开始提取特征...")
all_features, all_labels = [], []
with torch.no_grad():
    from utils.utils_weibo import clipdata2gpu
    for batch in tqdm(test_loader, desc="正在提取特征"):
        batch_data = clipdata2gpu(batch, use_cuda=(DEVICE.type == 'cuda'))
        features = model.extract_features(**batch_data)
        labels = batch_data['label']
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
logger.info(f"特征提取完成！共得到 {all_features.shape[0]} 个样本。")

# ============================ 导出特征代码块 (新添加) ============================
# 定义保存特征的目录和文件名
output_dir = './extracted_features'
os.makedirs(output_dir, exist_ok=True)
feature_filename = os.path.join(output_dir, f'features_{args.dataset}.npz')

# 使用 np.savez_compressed 保存特征和标签
logger.info(f"正在将提取的特征和标签保存至: {feature_filename}")
np.savez_compressed(feature_filename, features=all_features, labels=all_labels)
logger.info("特征和标签保存成功！")
# ==============================================================================

# --- 6. t-SNE 降维 ---
logger.info("开始执行 t-SNE 降维 (使用调整后的参数)...")
tsne = TSNE(
    n_components=2,
    verbose=1,
    perplexity=15,
    early_exaggeration=20,
    n_iter=700,
    learning_rate='auto',
    init='pca',
    random_state=42
)
tsne_results = tsne.fit_transform(all_features)
logger.info("t-SNE 降维完成！")

# --- 7. 绘图 ---
logger.info("正在绘制可视化图...")
fig, ax = plt.subplots(figsize=(8, 8))

custom_palette = ["#E67E22", "#3498DB"]  # 活力橙 vs 科技蓝

sns.scatterplot(
    x=tsne_results[:, 0],
    y=tsne_results[:, 1],
    hue=all_labels,
    palette=custom_palette,
    s=80,
    alpha=0.85,
    linewidth=0,
    legend=False,
    ax=ax
)

ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_ylabel(config['dataset'].capitalize(), rotation='vertical', fontsize=28, labelpad=20)

output_dir = './visualizations'
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, f'tsne_{config["dataset"]}_more_overlap.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
logger.info(f"可视化图已保存至: {output_filename}")
plt.show()