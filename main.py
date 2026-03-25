import os
import argparse
import logging
import random
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("runner")

# ---------------------------
# 1) Presets: Default values for the three datasets
# ---------------------------
PRESETS = {
    "gossipcop": {
        "model_name": "dccf_gossipcop",
        "model_py_path": "./model/dccf_gossipcop.py",
        "lr": 3e-4,
        "batchsize": 24,
        "seed": 2024,
        "early_stop": 100,
        "early_stop_metric": "acc",
        "data_dir_key": "gossipcop_data_dir",
    },
    "weibo21": {
        "model_name": "dccf_weibo21",
        "model_py_path": "./model/dccf_weibo21.py",
        "lr": 5e-4,
        "batchsize": 64,
        "seed": 3074,
        "early_stop": 100,
        "early_stop_metric": "F1",
        "data_dir_key": "weibo21_data_dir",
    },
    "weibo": {
        "model_name": "dccf_weibo",
        "model_py_path": "./model/dccf_weibo.py",
        "lr": 2e-4,
        "batchsize": 64,
        "seed": 3074,
        "early_stop": 100,
        "early_stop_metric": "F1",
        "data_dir_key": "weibo_data_dir",
    },
}

def pick(user_value, preset_value):
    return preset_value if user_value is None else user_value

# ---------------------------
# 2) Arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Unified runner for gossipcop/weibo21/weibo using DCCF framework")

parser.add_argument("--dataset", choices=["gossipcop", "weibo21", "weibo"], required=True,
                    help="Select the dataset to run.")

# General training hyperparameters (Default to None -> Use presets)
parser.add_argument("--model_name", default=None,
                    help="Model architecture name: dccf_gossipcop, dccf_weibo21, or dccf_weibo.")
parser.add_argument("--model_py_path", default=None,
                    help="Path to the model Python file.")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--batchsize", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--early_stop", type=int, default=None)
parser.add_argument("--early_stop_metric", choices=["acc", "F1"], default=None)

# Training & Environment
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--max_len", type=int, default=197)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--gpu", default="0")
parser.add_argument("--bert_emb_dim", type=int, default=768)
parser.add_argument("--save_param_dir", default="./param_model")
parser.add_argument("--emb_type", default="bert")

# DCCF-specific Hyperparameters
parser.add_argument("--lambda_F", type=float, default=0.075, 
                    help="Balancing coefficient for fact space auxiliary loss.")
parser.add_argument("--lambda_E", type=float, default=0.075, 
                    help="Balancing coefficient for sentiment space auxiliary loss.")
parser.add_argument("--M", type=int, default=4, 
                    help="Number of iterations for DARFU module.")
parser.add_argument("--tau", type=float, default=1.5, 
                    help="Temperature coefficient for discrepancy-aware weighting.")
parser.add_argument("--K_T", type=int, default=10, 
                    help="Number of textual semantic units.")
parser.add_argument("--K_I", type=int, default=10, 
                    help="Number of visual semantic units.")

# Pretrained Models and Data Paths
parser.add_argument("--bert_model_path_gossipcop", default="./pretrained_model/bert-base-uncased")
parser.add_argument("--clip_model_path_gossipcop", default="./pretrained_model/clip-vit-base-patch16")

parser.add_argument("--bert_model_path_weibo", default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch")
parser.add_argument("--bert_vocab_file_weibo", default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt")

parser.add_argument("--gossipcop_data_dir", default="./gossipcop/")
parser.add_argument("--weibo_data_dir", default="./data/")
parser.add_argument("--weibo21_data_dir", default="./Weibo_21/")

args = parser.parse_args()

# Set GPU visibility first
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

try:
    from run import Run  # Your training entry class
except Exception as e:
    logger.error(f"Failed to import Run: {e}. Please ensure run.py exists and implements Run(config).main().")
    raise SystemExit(1)

# ---------------------------
# 3) Apply presets & User overrides based on dataset
# ---------------------------
p = PRESETS[args.dataset]

current = {
    "dataset": args.dataset,
    "model_name": pick(args.model_name, p["model_name"]),
    "model_py_path": pick(args.model_py_path, p["model_py_path"]), 
    "lr": pick(args.lr, p["lr"]),
    "batchsize": pick(args.batchsize, p["batchsize"]),
    "seed": pick(args.seed, p["seed"]),
    "early_stop": pick(args.early_stop, p["early_stop"]),
    "early_stop_metric": pick(args.early_stop_metric, p["early_stop_metric"]),
}

# ---------------------------
# 4) Fix randomness for reproducibility
# ---------------------------
random.seed(current["seed"])
np.random.seed(current["seed"])
torch.manual_seed(current["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(current["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# 5) Assemble unified config (Consistent Run interface)
# ---------------------------
use_cuda = torch.cuda.is_available()
config = {
    "use_cuda": use_cuda,
    "dataset": current["dataset"],
    "model_name": current["model_name"],
    "model_py_path": current["model_py_path"], 

    # Data and model paths
    "gossipcop_data_dir": args.gossipcop_data_dir,
    "weibo_data_dir": args.weibo_data_dir,
    "weibo21_data_dir": args.weibo21_data_dir,

    "bert_model_path_gossipcop": args.bert_model_path_gossipcop,
    "clip_model_path_gossipcop": args.clip_model_path_gossipcop,

    "bert_model_path_weibo": args.bert_model_path_weibo,
    "bert_vocab_file_weibo": args.bert_vocab_file_weibo,

    # DCCF Hyperparameters
    "lambda_F": args.lambda_F,
    "lambda_E": args.lambda_E,
    "M": args.M,
    "tau": args.tau,
    "K_T": args.K_T,
    "K_I": args.K_I,

    # Training parameters
    "batchsize": current["batchsize"],
    "max_len": args.max_len,
    "early_stop": current["early_stop"],
    "early_stop_metric": current["early_stop_metric"],
    "num_workers": args.num_workers,
    "emb_type": args.emb_type,
    "weight_decay": 5e-5,
    "model_params": {"mlp": {"dims": [384], "dropout": 0.2}},
    "emb_dim": args.bert_emb_dim,
    "lr": current["lr"],
    "epoch": args.epoch,
    "seed": current["seed"],
    "save_param_dir": args.save_param_dir,
}

# Weibo/Weibo21 require vocab and specific bert model mapping
if args.dataset in {"weibo", "weibo21"}:
    config["vocab_file"] = args.bert_vocab_file_weibo
    config["bert"] = args.bert_model_path_weibo

# ---------------------------
# 6) Logging
# ---------------------------
logger.info("===== Final Config =====")
for k, v in config.items():
    logger.info(f"{k}: {v}")
logger.info("========================")

# ---------------------------
# 7) Execution
# ---------------------------
if __name__ == "__main__":
    if config["use_cuda"]:
        logger.info(f"CUDA is available. Using GPU {args.gpu}")
    else:
        logger.warning("CUDA is not available. Falling back to CPU.")

    # Note: The Run class internally needs to support dynamic loading 
    # of the model file via config["model_py_path"].
    runner = Run(config=config)
    runner.main()
    logger.info("Execution finished.")