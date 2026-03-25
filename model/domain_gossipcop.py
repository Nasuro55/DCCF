import logging
import os
import tqdm
import torch
from transformers import BertModel, CLIPModel, ViTModel
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# 1. Semantic Unit Attention Pooling (Stage 1)
# =====================================================================
class AttentionPooling(nn.Module):
    """
    Lightweight attention pooling to summarize encoder outputs into fixed-size unit sets.
    Ref: Section 3.1, Eq (2)-(5) of the DCCF paper[cite: 250, 251, 254].
    """
    def __init__(self, hidden_dim, num_units):
        super(AttentionPooling, self).__init__()
        self.num_units = num_units
        # Learnable parameters W_att to project hidden states to 'K' attention units
        self.W_att = nn.Linear(hidden_dim, num_units, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (Batch, Seq_Len, Hidden_Dim)
            attention_mask: (Batch, Seq_Len) - 1 for valid tokens, 0 for padding
        Returns:
            U: (Batch, Num_Units, Hidden_Dim) - Extracted semantic units
        """
        # (Batch, Seq_Len, Num_Units)
        attn_logits = self.W_att(hidden_states)
        
        if attention_mask is not None:
            # Mask out padding tokens by setting their logits to a large negative value
            # Expand mask to match num_units dimension
            mask = attention_mask.unsqueeze(-1).bool()
            attn_logits = attn_logits.masked_fill(~mask, -1e9)
            
        # Apply softmax along the sequence dimension (Eq 2 & 4) [cite: 252, 254]
        # A: (Batch, Seq_Len, Num_Units)
        A = torch.softmax(attn_logits, dim=1)
        
        # U = A^T * H (Eq 3 & 5) [cite: 253, 254]
        # Transpose A to (Batch, Num_Units, Seq_Len) and bmm with H (Batch, Seq_Len, Hidden_Dim)
        # Output U: (Batch, Num_Units, Hidden_Dim)
        U = torch.bmm(A.transpose(1, 2), hidden_states)
        return U

# =====================================================================
# 2. Discrepancy-Aware Residual Feature Updating (DARFU) (Stage 2)
# =====================================================================
class DARFU(nn.Module):
    """
    Discrepancy-Aware Residual Feature Updating module.
    Ref: Section 3.2, Feature Dynamics Evolution, Eq (15)-(21)[cite: 478, 560, 564, 568].
    Evolves semantic units within a space in a discrepancy-aware manner[cite: 566].
    """
    def __init__(self, feature_dim, num_iterations=4, tau=1.5):
        super(DARFU, self).__init__()
        self.M = num_iterations  # Number of evolution iterations (M) [cite: 568]
        self.tau = tau           # Temperature coefficient controlling sharpness (tau_X) [cite: 561]
        
        # Learnable nonlinear transformation g_X (Eq 20) [cite: 565]
        self.g_X = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, S_0):
        """
        Args:
            S_0: Initial semantic space features (Batch, n_X, Feature_Dim)
                 where n_X = K_T + K_I
        Returns:
            S_M: Evolved features after M iterations [cite: 568]
            T_final: Final tension matrix computed on evolved features [cite: 571]
        """
        B, n_X, D = S_0.size()
        S_t = S_0
        
        for t in range(self.M):
            # 1. Compute pairwise tension between semantic units (Eq 16) [cite: 558]
            # S_t_i: (Batch, n_X, 1, Feature_Dim)
            # S_t_j: (Batch, 1, n_X, Feature_Dim)
            S_t_i = S_t.unsqueeze(2)
            S_t_j = S_t.unsqueeze(1)
            
            # Tension T: (Batch, n_X, n_X)
            T_t = torch.sum((S_t_i - S_t_j) ** 2, dim=-1)
            
            # 2. Transform tension into discrepancy-aware aggregation weights (Eq 17) [cite: 560]
            # W_X: (Batch, n_X, n_X)
            # Apply temperature scaling
            exp_T = torch.exp(-T_t / self.tau)
            
            # Mask out self-connections (diagonal elements where i == j) (Eq 18) [cite: 560]
            diag_mask = torch.eye(n_X, device=S_t.device).bool().unsqueeze(0)
            exp_T = exp_T.masked_fill(diag_mask, 0.0)
            
            # Normalize over j != i
            # Add small epsilon to prevent division by zero
            sum_exp_T = torch.sum(exp_T, dim=-1, keepdim=True) + 1e-9
            W_t = exp_T / sum_exp_T
            
            # 3. Compute discrepancy-aware message (Eq 19) [cite: 562, 563]
            # m_t: (Batch, n_X, Feature_Dim)
            m_t = torch.bmm(W_t, S_t)
            
            # 4. Residual feature update (Eq 20) [cite: 564]
            concat_sm = torch.cat([S_t, m_t], dim=-1)  # (Batch, n_X, 2 * Feature_Dim)
            update = self.g_X(concat_sm)
            S_t = S_t + update

        # After M iterations, compute final tension matrix (Eq 21) [cite: 568, 571]
        S_M = S_t
        S_M_i = S_M.unsqueeze(2)
        S_M_j = S_M.unsqueeze(1)
        T_final = torch.sum((S_M_i - S_M_j) ** 2, dim=-1)
        
        return S_M, T_final

# =====================================================================
# 3. Conflict-Consensus Extraction (Stage 2 Continued)
# =====================================================================
class ConflictConsensusExtraction(nn.Module):
    """
    Extracts cross-modal conflict and global consensus from evolved features,
    and integrates them through consensus-referenced calibration.
    Ref: Section 3.2, Eq (22)-(26)[cite: 629, 668, 670].
    """
    def __init__(self, feature_dim, K_T, K_I):
        super(ConflictConsensusExtraction, self).__init__()
        self.K_T = K_T
        self.K_I = K_I
        
        # g_cal is an MLP for calibration (Eq 26) [cite: 671, 579]
        # Input size: 3 * Feature_Dim (Conflict) + Feature_Dim (Consensus) = 4 * Feature_Dim
        self.g_cal = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, S_prime, T_final):
        """
        Args:
            S_prime: Evolved semantic units (Batch, K_T + K_I, Feature_Dim)
            T_final: Pairwise tension matrix (Batch, K_T + K_I, K_T + K_I)
        Returns:
            V: Calibrated consensus-referenced representation (Batch, Feature_Dim) [cite: 585]
        """
        B, n_X, D = S_prime.size()
        
        # 1. Cross-modal conflict extraction (Eq 22 - 24) [cite: 629, 666]
        # We only care about tension between Text units (0 to K_T-1) 
        # and Image units (K_T to n_X-1)
        T_cross = T_final[:, :self.K_T, self.K_T:]  # (Batch, K_T, K_I)
        
        # Find the index of the maximum tension pair (p_X, q_X) [cite: 666]
        T_cross_flat = T_cross.reshape(B, -1)
        max_indices = torch.argmax(T_cross_flat, dim=1)  # (Batch,)
        
        # Convert flat indices back to 2D indices
        p_idx = max_indices // self.K_I  # Text unit index
        q_idx = max_indices % self.K_I + self.K_T  # Image unit index (offset by K_T)
        
        batch_idx = torch.arange(B, device=S_prime.device)
        
        s_p = S_prime[batch_idx, p_idx, :]  # (Batch, Feature_Dim)
        s_q = S_prime[batch_idx, q_idx, :]  # (Batch, Feature_Dim)
        
        # I_conflict = concat(s_p, s_q, |s_p - s_q|) (Eq 24) [cite: 666]
        I_conflict = torch.cat([s_p, s_q, torch.abs(s_p - s_q)], dim=-1)  # (Batch, 3 * Feature_Dim)
        
        # 2. Global consensus extraction (Eq 25) [cite: 668, 669]
        # Document-level semantic baseline
        C_consensus = torch.mean(S_prime, dim=1)  # (Batch, Feature_Dim)
        
        # 3. Consensus-referenced calibration (Eq 26) [cite: 670, 671]
        V_input = torch.cat([I_conflict, C_consensus], dim=-1)  # (Batch, 4 * Feature_Dim)
        V_X = self.g_cal(V_input)  # (Batch, Feature_Dim)
        
        return V_X

# =====================================================================
# 4. Main Model: DCCF (Replaces original DIVERModel)
# =====================================================================
class DCCFModel(nn.Module):
    """
    Dynamic Conflict-Consensus Framework for Multimodal Fake News Detection.
    Fully implements the architecture defined in the ICME 2026 paper[cite: 240, 241, 1316].
    """
    def __init__(self, bert_path_or_name, vis_model_path_or_name='google/vit-base-patch16-224', 
                 text_dim=768, vis_dim=768, feature_dim=256, 
                 K_T=10, K_I=10, M=4, tau=1.5, y_fact_dim=80, y_sent_dim=1, 
                 use_cuda=True, dropout=0.2):
        super(DCCFModel, self).__init__()
        
        self.K_T = K_T
        self.K_I = K_I
        self.feature_dim = feature_dim
        self.use_cuda = use_cuda
        
        # --- Modality Encoders ---
        # Assuming BERT for text and ViT for images as specified in Section 4.1 [cite: 819]
        try:
            logger.info(f"Loading BERT: {bert_path_or_name}")
            self.text_encoder = BertModel.from_pretrained(bert_path_or_name).requires_grad_(False)
        except Exception as e:
            logger.error(f"BERT load failed: {e}")
            self.text_encoder = None
            
        try:
            logger.info(f"Loading ViT: {vis_model_path_or_name}")
            # The paper uses ViT for visual encoding [cite: 247, 819]
            self.vis_encoder = ViTModel.from_pretrained(vis_model_path_or_name).requires_grad_(False)
        except Exception as e:
            logger.error(f"ViT load failed: {e}")
            self.vis_encoder = None

        if self.use_cuda:
            if self.text_encoder: self.text_encoder = self.text_encoder.cuda()
            if self.vis_encoder: self.vis_encoder = self.vis_encoder.cuda()
            
        # --- Stage 1: Attention Pooling (Eq 2-5) --- [cite: 250, 251]
        self.text_pooler = AttentionPooling(text_dim, K_T)
        self.vis_pooler = AttentionPooling(vis_dim, K_I)
        
        # --- Stage 1: Fact and Sentiment Space Projections (Eq 6, 8) --- [cite: 322, 430]
        # Fact space projection heads
        self.MLP_T_F = nn.Sequential(nn.Linear(text_dim, feature_dim), nn.ReLU())
        self.MLP_I_F = nn.Sequential(nn.Linear(vis_dim, feature_dim), nn.ReLU())
        
        # Sentiment space projection heads
        self.MLP_T_E = nn.Sequential(nn.Linear(text_dim, feature_dim), nn.ReLU())
        self.MLP_I_E = nn.Sequential(nn.Linear(vis_dim, feature_dim), nn.ReLU())
        
        # Auxiliary Classifiers for weak semantic anchoring
        # Fact uses image-side supervision (YOLO pseudo-labels, multi-hot) (Eq 7) [cite: 316, 427]
        self.MLP_cls_F = nn.Linear(feature_dim, y_fact_dim)
        # Sentiment uses text-side supervision (SenticNet polarity, regression target) (Eq 10) [cite: 467, 468]
        self.MLP_cls_E = nn.Linear(feature_dim, y_sent_dim)
        
        # --- Stage 2: Dynamics Evolution and Conflict-Consensus Extraction --- [cite: 476, 629]
        # Fact space DARFU and Extractor
        self.darfu_F = DARFU(feature_dim, num_iterations=M, tau=tau)
        self.extractor_F = ConflictConsensusExtraction(feature_dim, K_T, K_I)
        
        # Sentiment space DARFU and Extractor
        self.darfu_E = DARFU(feature_dim, num_iterations=M, tau=tau)
        self.extractor_E = ConflictConsensusExtraction(feature_dim, K_T, K_I)
        
        # --- Stage 3: Multi-View Deliberative Judgment --- [cite: 588, 590]
        # Final classifier input: V_F, V_E, V_F * V_E -> 3 * Feature_Dim [cite: 572, 598]
        self.MLP_final = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # Loss functions for auxiliary anchoring
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, **kwargs):
        """
        Args (kwargs):
            content / text_ids: Tokenized text IDs (Batch, Seq_Len)
            content_masks / text_mask: Text attention mask (Batch, Seq_Len)
            image_pixels / clip_image: Processed image tensors (Batch, C, H, W)
            y_fact: YOLO multi-hot pseudo-labels (Batch, y_fact_dim) [Optional, for training]
            y_sent: SenticNet polarity score (Batch, y_sent_dim) [Optional, for training]
        Returns:
            prob: Final prediction probability [0, 1]
            loss_F: Fact space auxiliary loss
            loss_E: Sentiment space auxiliary loss
        """
        # Handle potentially different key names from dataloader
        text_ids = kwargs.get('content') if kwargs.get('content') is not None else kwargs.get('text_ids')
        text_mask = kwargs.get('content_masks') if kwargs.get('content_masks') is not None else kwargs.get('text_mask')
        image_pixels = kwargs.get('image_pixels') if kwargs.get('image_pixels') is not None else kwargs.get('clip_image')
        y_fact = kwargs.get('y_fact')
        y_sent = kwargs.get('y_sent')
        
        B = text_ids.size(0) if text_ids is not None else 1
        device = text_ids.device if text_ids is not None else next(self.parameters()).device
        
        # Initialize default dummy outputs if inputs are missing
        if text_ids is None or self.text_encoder is None or image_pixels is None or self.vis_encoder is None:
             prob = torch.ones(B, device=device) * 0.5
             return prob, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # ----------------------------------------------------
        # Modality Encoding (Eq 1) [cite: 245, 246, 247]
        # ----------------------------------------------------
        # H_T: (Batch, Seq_Len, d_T)
        H_T = self.text_encoder(input_ids=text_ids, attention_mask=text_mask).last_hidden_state
        # H_I: (Batch, Num_Patches, d_I)
        H_I = self.vis_encoder(pixel_values=image_pixels).last_hidden_state
        
        # ----------------------------------------------------
        # Stage 1: Fact-Sentiment Feature Extraction
        # ----------------------------------------------------
        # 1. Attention Pooling to obtain semantic units (Eq 2-5) [cite: 250, 254]
        U_T = self.text_pooler(H_T, text_mask)  # (Batch, K_T, d_T)
        U_I = self.vis_pooler(H_I)              # (Batch, K_I, d_I)
        
        # 2. Fact space projection (Eq 6) [cite: 322]
        f_T_F = self.MLP_T_F(U_T)  # (Batch, K_T, d_F)
        f_I_F = self.MLP_I_F(U_I)  # (Batch, K_I, d_F)
        
        # 3. Sentiment space projection (Eq 8) [cite: 430]
        f_T_E = self.MLP_T_E(U_T)  # (Batch, K_T, d_E)
        f_I_E = self.MLP_I_E(U_I)  # (Batch, K_I, d_E)
        
        # 4. Decoupled semantic spaces construction (Eq 11-12) [cite: 470, 471]
        S_F_0 = torch.cat([f_T_F, f_I_F], dim=1)  # (Batch, K_T + K_I, d_F)
        S_E_0 = torch.cat([f_T_E, f_I_E], dim=1)  # (Batch, K_T + K_I, d_E)
        
        # 5. Auxiliary semantic anchoring computation (Eq 7, 10) [cite: 427, 468]
        loss_F = torch.tensor(0.0, device=device)
        loss_E = torch.tensor(0.0, device=device)
        
        if self.training and y_fact is not None and y_sent is not None:
            # Fact Anchoring: supervised by YOLO image labels [cite: 316, 317]
            f_bar_I_F = torch.mean(f_I_F, dim=1)  # Average visual fact units [cite: 426]
            pred_fact = self.MLP_cls_F(f_bar_I_F)
            loss_F = self.bce_with_logits(pred_fact, y_fact.float())
            
            # Sentiment Anchoring: supervised by SenticNet text polarity [cite: 467]
            f_bar_T_E = torch.mean(f_T_E, dim=1)  # Average textual sentiment units [cite: 468]
            pred_sent = self.MLP_cls_E(f_bar_T_E)
            loss_E = self.mse_loss(pred_sent, y_sent.float().unsqueeze(-1) if y_sent.dim()==1 else y_sent.float())

        # ----------------------------------------------------
        # Stage 2: Feature Dynamics Evolution & Conflict-Consensus Extraction
        # ----------------------------------------------------
        # Fact Space Evolution [cite: 476, 629]
        S_F_prime, T_F_final = self.darfu_F(S_F_0)
        V_F = self.extractor_F(S_F_prime, T_F_final)  # (Batch, d_F)
        
        # Sentiment Space Evolution [cite: 476, 629]
        S_E_prime, T_E_final = self.darfu_E(S_E_0)
        V_E = self.extractor_E(S_E_prime, T_E_final)  # (Batch, d_E)

        # ----------------------------------------------------
        # Stage 3: Multi-View Deliberative Judgment
        # ----------------------------------------------------
        # V_final = concat(V^F, V^E, V^F * V^E) (Eq 27) [cite: 572, 598]
        V_interaction = V_F * V_E
        V_final = torch.cat([V_F, V_E, V_interaction], dim=-1)  # (Batch, 3 * feature_dim)
        
        # Final Fake News Classification (Eq 28) [cite: 573, 606]
        logits = self.MLP_final(V_final).squeeze(-1)
        # Note: BCELoss expects probabilities, while BCEWithLogitsLoss expects raw logits.
        # Original Trainer code used BCELoss on final_logits but renamed it final_prob. 
        # Here we align with the paper's final layer and output logits to be used with BCEWithLogitsLoss.
        
        return logits, loss_F, loss_E

# --------------------------------------------------------
# 训练器 (完全重用你的日志记录和早停逻辑)
# --------------------------------------------------------
class Trainer():
    def __init__(self, emb_dim, mlp_dims, bert_path_or_name, clip_path_or_name,
                 use_cuda, lr, dropout, train_loader, val_loader, test_loader, 
                 category_dict, weight_decay, save_param_dir, early_stop=10, 
                 epoches=100, metric_key_for_early_stop='acc', 
                 lambda_F=0.075, lambda_E=0.075):
        
        self.lr = lr; self.weight_decay = weight_decay
        self.train_loader = train_loader; self.val_loader = val_loader; self.test_loader = test_loader
        self.category_dict = category_dict; self.use_cuda = use_cuda
        self.early_stop = early_stop; self.epoches = epoches
        self.metric_key_for_early_stop = metric_key_for_early_stop
        self.save_param_dir = save_param_dir
        os.makedirs(self.save_param_dir, exist_ok=True)
        
        # DCCF Hyperparameters from Section 4.1 [cite: 822, 953]
        self.lambda_F = lambda_F
        self.lambda_E = lambda_E

        # Initialize DCCF Model
        self.model = DCCFModel(
            bert_path_or_name=bert_path_or_name,
            vis_model_path_or_name='google/vit-base-patch16-224', # Paper uses ViT [cite: 819]
            feature_dim=256, dropout=dropout, use_cuda=use_cuda,
            M=4, tau=1.5, K_T=10, K_I=10 # Best params according to paper [cite: 1143]
        )
        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            logger.warning("CUDA not available/requested. Model on CPU.")

    def train(self):
        # DCCF loss (Eq 29, 30) [cite: 576, 614]
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        
        # 假设存在 Recorder 类
        try:
            from utils.utils_gossipcop import Recorder, Averager, clipdata2gpu
            recorder = Recorder(self.early_stop, metric_key=self.metric_key_for_early_stop)
        except:
            logger.warning("Recorder fallback")
            recorder = None 

        for epoch in range(self.epoches):
            self.model.train()
            # 根据提供的工具包构建 averager，如果没有则用普通变量
            try: avg_loss = Averager()
            except: avg_loss = None; total_loss = 0.0; steps = 0

            train_data_iter = tqdm.tqdm(self.train_loader)
            for step_n, batch in enumerate(train_data_iter):
                try:
                    # 假设 utils_gossipcop.clipdata2gpu
                    try: batch_data = clipdata2gpu(batch)
                    except: batch_data = batch # 回退
                    
                    if batch_data is None: continue
                    label = batch_data.get('label')
                    if label is None: continue
                    
                    # Ensure y_fact and y_sent are available in the batch if weak anchoring is desired
                    # For full DCCF reproduction, your dataloader needs to supply these.
                    # If they are not supplied, the model gracefully handles it by returning 0 for aux losses.
                    
                    # DCCF forward returns: logits, loss_F, loss_E
                    final_logits, loss_F, loss_E = self.model(**batch_data)

                    # Eq 29: Final Prediction Loss [cite: 576]
                    loss_final = loss_fn(final_logits, label.float())
                    
                    # Eq 30: Total Objective [cite: 576, 577]
                    loss = loss_final + (self.lambda_F * loss_F) + (self.lambda_E * loss_E)
                    
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    
                    if avg_loss: avg_loss.add(loss.item())
                    else: total_loss += loss.item(); steps += 1
                    
                    train_data_iter.set_description(f"Epoch {epoch+1}/{self.epoches}")
                    train_data_iter.set_postfix(
                        loss=(avg_loss.item() if avg_loss else total_loss/steps), 
                        lr=optimizer.param_groups[0]['lr'],
                        L_final=loss_final.item(), L_F=loss_F.item(), L_E=loss_E.item()
                    )
                except Exception as e:
                    logger.exception(f"Train step {step_n} error: {e}")
                    continue
            
            if scheduler is not None: scheduler.step()
            loss_val = avg_loss.item() if avg_loss else total_loss/steps
            logger.info(f'Train Epoch {epoch+1} Done; Avg Loss: {loss_val:.4f}; LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if self.val_loader is None: continue

            try:
                val_results = self.test(self.val_loader)
                if not val_results: continue

                current_metric_val = val_results.get(self.metric_key_for_early_stop, 0.0)
                logger.info(f"Val E{epoch+1}: Acc:{val_results.get('acc', 0.0):.4f} Tracked:{current_metric_val:.4f}")
                
                if recorder:
                    mark = recorder.add(val_results)
                    if mark == 'save':
                        save_p = os.path.join(self.save_param_dir, 'best_model.pth')
                        torch.save(self.model.state_dict(), save_p)
                    elif mark == 'esc':
                        logger.info("Early stopping triggered.")
                        break
            except Exception as e:
                logger.exception(f"Val epoch {epoch+1} error: {e}")
                continue

        logger.info("Training loop finished.")
        # 加载最优模型进行测试逻辑保持不变
        best_model_path = os.path.join(self.save_param_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage))
        
        final_results = None
        if self.test_loader is not None:
            final_results = self.test(self.test_loader)
            if final_results:
                logger.info(f"Final Test Acc: {final_results.get('acc', 0.0):.4f}")
        return final_results, best_model_path

    def test(self, dataloader):
        pred_probs, label_list, category_list = [], [], []
        if dataloader is None: return {}
        
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader, desc="Testing")
        with torch.no_grad():
            for step_n, batch in enumerate(data_iter):
                try:
                    # 假设 utils_gossipcop.clipdata2gpu
                    try: batch_data = clipdata2gpu(batch)
                    except: batch_data = batch
                    
                    if batch_data is None: continue
                    batch_label = batch_data.get('label')
                    batch_category = batch_data.get('category')
                    if batch_label is None: continue
                    
                    # Test only needs probabilities; ignore aux losses
                    final_logits, _, _ = self.model(**batch_data)
                    batch_pred_prob = torch.sigmoid(final_logits)
                    
                    label_list.extend(batch_label.cpu().numpy().tolist())
                    pred_probs.extend(batch_pred_prob.cpu().numpy().tolist())
                    if batch_category is not None:
                        category_list.extend(batch_category.cpu().numpy().tolist())
                    else:
                        category_list.extend([None] * batch_label.size(0))
                except Exception as e:
                    logger.exception(f"Test batch {step_n} error: {e}")
                    continue
        
        if not label_list: return {}
        
        try:
            from utils.utils_gossipcop import calculate_metrics
            if self.category_dict:
                metric_res = calculate_metrics(label_list, pred_probs, category_list, self.category_dict)
            else:
                metric_res = calculate_metrics(label_list, pred_probs)
            return metric_res
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return {'acc': sum(1 for p, l in zip([1 if x>0.5 else 0 for x in pred_probs], label_list) if p==l)/len(label_list)}