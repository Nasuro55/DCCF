import os
import logging
import traceback
import inspect
import importlib
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPProcessor

# -----------------------
# Logger Configuration
# -----------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# -----------------------
# Optional Dataset Imports
# -----------------------
FakeNet_dataset = None
try:
    from FakeNet_dataset import FakeNet_dataset
    logger.info("FakeNet_dataset imported successfully.")
except Exception as e:
    logger.warning(f"FakeNet_dataset not imported: {e}")

WeiboDataLoaderClass = None
try:
    from utils.clip_dataloader import bert_data as WeiboDataLoaderClass
    logger.info("utils.clip_dataloader.bert_data (Weibo) imported successfully.")
except Exception as e:
    logger.warning(f"Weibo dataloader not imported: {e}")

Weibo21DataLoaderClass = None
try:
    from utils.weibo21_clip_dataloader import bert_data as Weibo21DataLoaderClass
    logger.info("utils.weibo21_clip_dataloader.bert_data (Weibo21) imported successfully.")
except Exception as e:
    logger.warning(f"Weibo21 dataloader not imported: {e}")

# -----------------------
# Global Tokenizer/Processor for GossipCop
# -----------------------
bert_tokenizer_gossipcop = None
clip_processor_gossipcop = None

# -----------------------
# Collate Function for GossipCop
# -----------------------
def collate_fn_gossipcop(batch):
    """Filters None values in batch; automatically stacks tensor fields; keeps other fields as list."""
    original_len = len(batch)
    batch = [x for x in batch if x is not None]
    
    if original_len > 0 and not batch:
        logger.warning("Collate(gossipcop): All samples are None. Returning None.")
        return None
    if not batch:
        return None

    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [it[k] for it in batch]
        if all(isinstance(v, torch.Tensor) for v in vals):
            try:
                out[k] = torch.stack(vals, dim=0)
            except RuntimeError as e:
                logger.error(f"Collate failed to stack key '{k}': {e}")
                for i, v in enumerate(vals):
                    logger.error(f"  Item #{i} shape={getattr(v, 'shape', None)}")
                return None
        else:
            out[k] = vals
    return out

# -----------------------
# Unified Runner Class
# -----------------------
class Run:
    """
    Unified Runner for DCCF Framework.
    Requires DCCF-specific hyperparameters in config:
      - lambda_F
      - lambda_E
      - M
      - tau
      - K_T
      - K_I
    """
    def __init__(self, config):
        self.config = config
        self.use_cuda = config.get("use_cuda", torch.cuda.is_available())

        # General configurations
        self.dataset = config["dataset"]
        self.model_name = config["model_name"]
        self.lr = config["lr"]
        self.batchsize = config["batchsize"]
        self.emb_dim = config["emb_dim"]
        self.max_len = config["max_len"]
        self.num_workers = config.get("num_workers", 0)
        self.early_stop = config["early_stop"]
        self.epoch = config["epoch"]
        self.save_param_dir = config["save_param_dir"]
        
        # DCCF Specific configurations
        self.lambda_F = config.get("lambda_F", 0.075)
        self.lambda_E = config.get("lambda_E", 0.075)
        self.M = config.get("M", 4)
        self.tau = config.get("tau", 1.5)
        self.K_T = config.get("K_T", 10)
        self.K_I = config.get("K_I", 10)

        logger.info(f"Run initialized. Dataset: {self.dataset}, Model: {self.model_name}")

        # Dataset specific initialization
        if self.dataset == "gossipcop":
            self._init_gossipcop()
        elif self.dataset == "weibo":
            self._init_weibo()
        elif self.dataset == "weibo21":
            self._init_weibo21()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
            
        # Dynamically load the Trainer class based on model_name
        self._load_trainer_class()

    def _load_trainer_class(self):
        """Dynamically imports the Trainer class based on self.model_name."""
        try:
            module_path = f"model.{self.model_name}"
            module = importlib.import_module(module_path)
            
            # Use 'Trainer' for gossipcop or 'DOMAINTrainerWeibo' for weibo/weibo21 based on previous logic
            if self.dataset == "gossipcop":
                self.TrainerClass = getattr(module, "Trainer")
            else:
                self.TrainerClass = getattr(module, "DOMAINTrainerWeibo", getattr(module, "Trainer", None))
                
            if self.TrainerClass is None:
                raise AttributeError(f"Could not find a valid Trainer class in {module_path}")
                
            logger.info(f"Successfully loaded Trainer from {module_path}")
        except Exception as e:
            logger.error(f"Failed to dynamically import Trainer for {self.model_name}: \n{traceback.format_exc()}")
            raise

    # -------- GossipCop Initialization --------
    def _init_gossipcop(self):
        self.root_path = self.config.get("gossipcop_data_dir")
        self.bert_model_path = self.config.get("bert_model_path_gossipcop")
        self.clip_model_path = self.config.get("clip_model_path_gossipcop") # Kept for dataloader processor
        self.category_dict = {"gossip": 0}
        self.early_stop_metric_key = self.config.get("early_stop_metric", "acc")

        for key, val in {
            "gossipcop_data_dir": self.root_path,
            "bert_model_path_gossipcop": self.bert_model_path,
            "clip_model_path_gossipcop": self.clip_model_path,
        }.items():
            if not val:
                raise ValueError(f"GossipCop missing configuration: {key}")

        global bert_tokenizer_gossipcop, clip_processor_gossipcop
        try:
            bert_tokenizer_gossipcop = BertTokenizer.from_pretrained(self.bert_model_path)
        except Exception as e:
            logger.error(f"Failed to load BERT tokenizer (GossipCop): {e}")
            
        try:
            clip_processor_gossipcop = CLIPProcessor.from_pretrained(self.clip_model_path)
        except Exception as e:
            logger.error(f"Failed to load CLIP processor (GossipCop): {e}")

        if bert_tokenizer_gossipcop is None or clip_processor_gossipcop is None:
            raise RuntimeError("Required tokenizer/processor for GossipCop failed to load.")

    # -------- Weibo Initialization --------
    def _init_weibo(self):
        self.root_path = self.config.get("weibo_data_dir")
        if not self.root_path:
            raise ValueError("Weibo missing configuration: weibo_data_dir")
            
        self.train_path = os.path.join(self.root_path, "train_origin.csv")
        self.val_path = os.path.join(self.root_path, "val_origin.csv")
        self.test_path = os.path.join(self.root_path, "test_origin.csv")

        self.category_dict = {"经济":0, "健康":1, "军事":2, "科学":3, "政治":4, "国际":5, "教育":6, "娱乐":7, "社会":8}
        self.bert_model_path = self.config.get("bert_model_path_weibo")
        self.vocab_file = self.config.get("vocab_file")
        self.early_stop_metric_key = self.config.get("early_stop_metric", "F1")
        
        if not self.bert_model_path or not self.vocab_file:
            raise ValueError("Weibo missing configuration: bert_model_path_weibo or vocab_file")

    # -------- Weibo21 Initialization --------
    def _init_weibo21(self):
        self.root_path = self.config.get("weibo21_data_dir")
        if not self.root_path:
            raise ValueError("Weibo21 missing configuration: weibo21_data_dir")
            
        self.train_path = os.path.join(self.root_path, "train_datasets.xlsx")
        self.val_path = os.path.join(self.root_path, "val_datasets.xlsx")
        self.test_path = os.path.join(self.root_path, "test_datasets.xlsx")

        self.category_dict = {"科技":0, "军事":1, "教育考试":2, "灾难事故":3, "政治":4, "医药健康":5, "财经商业":6, "文体娱乐":7, "社会生活":8}
        self.bert_model_path = self.config.get("bert_model_path_weibo")
        self.vocab_file = self.config.get("vocab_file")
        self.early_stop_metric_key = self.config.get("early_stop_metric", "F1")
        
        if not self.bert_model_path or not self.vocab_file:
            raise ValueError("Weibo21 missing configuration: bert_model_path_weibo or vocab_file")

    # -----------------------
    # Dataloaders Setup
    # -----------------------
    def get_dataloader(self):
        logger.info(f"Preparing dataloaders for dataset: {self.dataset}")
        
        if self.dataset == "gossipcop":
            if FakeNet_dataset is None:
                raise ImportError("FakeNet_dataset is missing.")
            img_size, clip_max_len = 224, 77

            train_ds = FakeNet_dataset(
                root_path=self.root_path,
                bert_tokenizer_instance=bert_tokenizer_gossipcop,
                clip_processor_instance=clip_processor_gossipcop,
                dataset_name="gossip",
                image_size=img_size,
                is_train=True,
                bert_max_len=self.max_len,
                clip_max_len=clip_max_len
            )
            val_ds = FakeNet_dataset(
                root_path=self.root_path,
                bert_tokenizer_instance=bert_tokenizer_gossipcop,
                clip_processor_instance=clip_processor_gossipcop,
                dataset_name="gossip",
                image_size=img_size,
                is_train=False,
                bert_max_len=self.max_len,
                clip_max_len=clip_max_len
            )
            train_loader = DataLoader(train_ds, batch_size=self.batchsize, shuffle=True,
                                      collate_fn=collate_fn_gossipcop, num_workers=self.num_workers,
                                      drop_last=True, pin_memory=self.use_cuda)
            val_loader = DataLoader(val_ds, batch_size=self.batchsize, shuffle=False,
                                    collate_fn=collate_fn_gossipcop, num_workers=self.num_workers,
                                    drop_last=False, pin_memory=self.use_cuda)
            test_loader = val_loader

        elif self.dataset == "weibo":
            if WeiboDataLoaderClass is None:
                raise ImportError("Missing utils.clip_dataloader.bert_data")
            loader = WeiboDataLoaderClass(
                max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                category_dict=self.category_dict, num_workers=self.num_workers,
                clip_model_name="ViT-B-16", clip_download_root="./"
            )
            train_loader = loader.load_data(
                self.train_path,
                os.path.join(self.root_path, "train_loader.pkl"),
                os.path.join(self.root_path, "train_clip_loader.pkl"),
                True
            )
            val_loader = loader.load_data(
                self.val_path,
                os.path.join(self.root_path, "val_loader.pkl"),
                os.path.join(self.root_path, "val_clip_loader.pkl"),
                False
            )
            test_loader = loader.load_data(
                self.test_path,
                os.path.join(self.root_path, "test_loader.pkl"),
                os.path.join(self.root_path, "test_clip_loader.pkl"),
                False
            )

        elif self.dataset == "weibo21":
            if Weibo21DataLoaderClass is None:
                raise ImportError("Missing utils.weibo21_clip_dataloader.bert_data")
            loader = Weibo21DataLoaderClass(
                max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                category_dict=self.category_dict, num_workers=self.num_workers
            )
            train_loader = loader.load_data(
                self.train_path,
                os.path.join(self.root_path, "train_loader.pkl"),
                os.path.join(self.root_path, "train_clip_loader.pkl"),
                True
            )
            val_loader = loader.load_data(
                self.val_path,
                os.path.join(self.root_path, "val_loader.pkl"),
                os.path.join(self.root_path, "val_clip_loader.pkl"),
                False
            )
            test_loader = loader.load_data(
                self.test_path,
                os.path.join(self.root_path, "test_loader.pkl"),
                os.path.join(self.root_path, "test_clip_loader.pkl"),
                False
            )
        else:
            raise ValueError(f"Dataloader not implemented for dataset: {self.dataset}")

        if not all([train_loader, val_loader, test_loader]):
            missing = [n for n, l in [("train", train_loader), ("val", val_loader), ("test", test_loader)] if l is None]
            raise RuntimeError(f"Dataloader initialization failed for: {', '.join(missing)}")

        return train_loader, val_loader, test_loader

    # -----------------------
    # Main Execution
    # -----------------------
    def main(self):
        try:
            train_loader, val_loader, test_loader = self.get_dataloader()
        except Exception:
            logger.error("Failed to acquire dataloaders: \n" + traceback.format_exc())
            return

        save_dir = os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Model parameters will be saved to: {save_dir}")

        mp = self.config.get("model_params", {}).get("mlp", {})
        
        # Unified parameters passed to the DCCF Trainer
        trainer_kwargs = dict(
            emb_dim=self.emb_dim,
            mlp_dims=mp.get("dims", [384]),
            use_cuda=self.use_cuda,
            lr=self.lr,
            dropout=mp.get("dropout", 0.2),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            category_dict=self.category_dict,
            weight_decay=self.config.get("weight_decay", 5e-5),
            save_param_dir=save_dir,
            early_stop=self.early_stop,
            metric_key_for_early_stop=self.early_stop_metric_key,
            epoches=self.epoch,
            # DCCF Core hyperparameters injection
            lambda_F=self.lambda_F,
            lambda_E=self.lambda_E,
            M=self.M,
            tau=self.tau,
            K_T=self.K_T,
            K_I=self.K_I
        )
        
        # Add dataset-specific path parameters
        if self.dataset == "gossipcop":
            trainer_kwargs.update({
                "bert_path_or_name": self.bert_model_path,
                "clip_path_or_name": getattr(self, "clip_model_path", None), # Kept for signature compatibility
            })
        else:
            trainer_kwargs.update({
                "bert_path_or_name": self.bert_model_path,
                "clip_path_or_name": getattr(self, "clip_model_path", None),
            })

        # Filter kwargs to only those accepted by the Trainer's __init__ to avoid crashes
        sig = inspect.signature(self.TrainerClass.__init__)
        valid_kwargs = {k: v for k, v in trainer_kwargs.items() if k in sig.parameters}
        
        try:
            trainer = self.TrainerClass(**valid_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize Trainer class: \n{traceback.format_exc()}")
            return

        logger.info("Trainer initialized successfully. Commencing training phase...")
        try:
            trainer.train()
            logger.info(f"Training completed successfully for: {self.dataset} - {self.model_name}")
        except Exception:
            logger.error("An exception occurred during training: \n" + traceback.format_exc())

# Code block for direct execution guidance
if __name__ == "__main__":
    logger.info("This file is meant to be imported and executed via main.py: Run(config).main()")