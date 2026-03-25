import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, ViTModel

# =====================================================================
# 1. Semantic Unit Attention Pooling (Stage 1)
# =====================================================================
class AttentionPooling(nn.Module):
    """
    Lightweight attention pooling to summarize encoder outputs into fixed-size unit sets.
    Ref: Section 3.1, Eq (2)-(5) of the DCCF paper.
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
            
        # Apply softmax along the sequence dimension (Eq 2 & 4)
        # A: (Batch, Seq_Len, Num_Units)
        A = torch.softmax(attn_logits, dim=1)
        
        # U = A^T * H (Eq 3 & 5)
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
    Ref: Section 3.2, Feature Dynamics Evolution, Eq (15)-(21).
    Evolves semantic units within a space in a discrepancy-aware manner.
    """
    def __init__(self, feature_dim, num_iterations=4, tau=1.5):
        super(DARFU, self).__init__()
        self.M = num_iterations  # Number of evolution iterations (M)
        self.tau = tau           # Temperature coefficient controlling sharpness (tau_X)
        
        # Learnable nonlinear transformation g_X (Eq 20)
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
            S_M: Evolved features after M iterations
            T_final: Final tension matrix computed on evolved features
        """
        B, n_X, D = S_0.size()
        S_t = S_0
        
        for t in range(self.M):
            # 1. Compute pairwise tension between semantic units (Eq 16)
            # S_t_i: (Batch, n_X, 1, Feature_Dim)
            # S_t_j: (Batch, 1, n_X, Feature_Dim)
            S_t_i = S_t.unsqueeze(2)
            S_t_j = S_t.unsqueeze(1)
            
            # Tension T: (Batch, n_X, n_X)
            T_t = torch.sum((S_t_i - S_t_j) ** 2, dim=-1)
            
            # 2. Transform tension into discrepancy-aware aggregation weights (Eq 17)
            # W_X: (Batch, n_X, n_X)
            # Apply temperature scaling
            exp_T = torch.exp(-T_t / self.tau)
            
            # Mask out self-connections (diagonal elements where i == j) (Eq 18)
            diag_mask = torch.eye(n_X, device=S_t.device).bool().unsqueeze(0)
            exp_T = exp_T.masked_fill(diag_mask, 0.0)
            
            # Normalize over j != i
            # Add small epsilon to prevent division by zero
            sum_exp_T = torch.sum(exp_T, dim=-1, keepdim=True) + 1e-9
            W_t = exp_T / sum_exp_T
            
            # 3. Compute discrepancy-aware message (Eq 19)
            # m_t: (Batch, n_X, Feature_Dim)
            m_t = torch.bmm(W_t, S_t)
            
            # 4. Residual feature update (Eq 20)
            concat_sm = torch.cat([S_t, m_t], dim=-1)  # (Batch, n_X, 2 * Feature_Dim)
            update = self.g_X(concat_sm)
            S_t = S_t + update

        # After M iterations, compute final tension matrix (Eq 21)
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
    Ref: Section 3.2, Eq (22)-(26).
    """
    def __init__(self, feature_dim, K_T, K_I):
        super(ConflictConsensusExtraction, self).__init__()
        self.K_T = K_T
        self.K_I = K_I
        
        # g_cal is an MLP for calibration (Eq 26)
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
            V: Calibrated consensus-referenced representation (Batch, Feature_Dim)
        """
        B, n_X, D = S_prime.size()
        
        # 1. Cross-modal conflict extraction (Eq 22 - 24)
        # We only care about tension between Text units (0 to K_T-1) 
        # and Image units (K_T to n_X-1)
        T_cross = T_final[:, :self.K_T, self.K_T:]  # (Batch, K_T, K_I)
        
        # Find the index of the maximum tension pair (p_X, q_X)
        T_cross_flat = T_cross.reshape(B, -1)
        max_indices = torch.argmax(T_cross_flat, dim=1)  # (Batch,)
        
        # Convert flat indices back to 2D indices
        p_idx = max_indices // self.K_I  # Text unit index
        q_idx = max_indices % self.K_I + self.K_T  # Image unit index (offset by K_T)
        
        batch_idx = torch.arange(B, device=S_prime.device)
        
        s_p = S_prime[batch_idx, p_idx, :]  # (Batch, Feature_Dim)
        s_q = S_prime[batch_idx, q_idx, :]  # (Batch, Feature_Dim)
        
        # I_conflict = concat(s_p, s_q, |s_p - s_q|) (Eq 24)
        I_conflict = torch.cat([s_p, s_q, torch.abs(s_p - s_q)], dim=-1)  # (Batch, 3 * Feature_Dim)
        
        # 2. Global consensus extraction (Eq 25)
        # Document-level semantic baseline
        C_consensus = torch.mean(S_prime, dim=1)  # (Batch, Feature_Dim)
        
        # 3. Consensus-referenced calibration (Eq 26)
        V_input = torch.cat([I_conflict, C_consensus], dim=-1)  # (Batch, 4 * Feature_Dim)
        V_X = self.g_cal(V_input)  # (Batch, Feature_Dim)
        
        return V_X

# =====================================================================
# 4. Main Model: DCCF (Replaces original DIVERModel)
# =====================================================================
class DCCFModel(nn.Module):
    """
    Dynamic Conflict-Consensus Framework for Multimodal Fake News Detection.
    Fully implements the architecture defined in the ICME 2026 paper.
    """
    def __init__(self, text_dim=768, vis_dim=768, feature_dim=256, 
                 K_T=10, K_I=10, M=4, tau=1.5, y_fact_dim=80, y_sent_dim=1):
        super(DCCFModel, self).__init__()
        
        self.K_T = K_T
        self.K_I = K_I
        self.feature_dim = feature_dim
        
        # --- Modality Encoders ---
        # Assuming BERT for text and ViT for images as specified in Section 4.1
        self.text_encoder = BertModel.from_pretrained('bert-base-chinese')
        self.vis_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Freeze encoders to save memory/computation if desired (optional)
        for param in self.text_encoder.parameters(): param.requires_grad = False
        for param in self.vis_encoder.parameters(): param.requires_grad = False
            
        # --- Stage 1: Attention Pooling (Eq 2-5) ---
        self.text_pooler = AttentionPooling(text_dim, K_T)
        self.vis_pooler = AttentionPooling(vis_dim, K_I)
        
        # --- Stage 1: Fact and Sentiment Space Projections (Eq 6, 8) ---
        # Fact space projection heads
        self.MLP_T_F = nn.Sequential(nn.Linear(text_dim, feature_dim), nn.ReLU())
        self.MLP_I_F = nn.Sequential(nn.Linear(vis_dim, feature_dim), nn.ReLU())
        
        # Sentiment space projection heads
        self.MLP_T_E = nn.Sequential(nn.Linear(text_dim, feature_dim), nn.ReLU())
        self.MLP_I_E = nn.Sequential(nn.Linear(vis_dim, feature_dim), nn.ReLU())
        
        # Auxiliary Classifiers for weak semantic anchoring
        # Fact uses image-side supervision (YOLO pseudo-labels, multi-hot) (Eq 7)
        self.MLP_cls_F = nn.Linear(feature_dim, y_fact_dim)
        # Sentiment uses text-side supervision (SenticNet polarity, regression target) (Eq 10)
        self.MLP_cls_E = nn.Linear(feature_dim, y_sent_dim)
        
        # --- Stage 2: Dynamics Evolution and Conflict-Consensus Extraction ---
        # Fact space DARFU and Extractor
        self.darfu_F = DARFU(feature_dim, num_iterations=M, tau=tau)
        self.extractor_F = ConflictConsensusExtraction(feature_dim, K_T, K_I)
        
        # Sentiment space DARFU and Extractor
        self.darfu_E = DARFU(feature_dim, num_iterations=M, tau=tau)
        self.extractor_E = ConflictConsensusExtraction(feature_dim, K_T, K_I)
        
        # --- Stage 3: Multi-View Deliberative Judgment ---
        # Final classifier input: V_F, V_E, V_F * V_E -> 3 * Feature_Dim
        self.MLP_final = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # Loss functions for auxiliary anchoring
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, text_ids, text_mask, image_pixels, y_fact=None, y_sent=None, **kwargs):
        """
        Args:
            text_ids: Tokenized text IDs (Batch, Seq_Len)
            text_mask: Text attention mask (Batch, Seq_Len)
            image_pixels: Processed image tensors (Batch, C, H, W)
            y_fact: YOLO multi-hot pseudo-labels (Batch, y_fact_dim)
            y_sent: SenticNet polarity score (Batch, y_sent_dim)
        Returns:
            prob: Final prediction probability [0, 1]
            loss_F: Fact space auxiliary loss
            loss_E: Sentiment space auxiliary loss
        """
        B = text_ids.size(0)
        
        # ----------------------------------------------------
        # Modality Encoding
        # ----------------------------------------------------
        # H_T: (Batch, Seq_Len, d_T)
        H_T = self.text_encoder(input_ids=text_ids, attention_mask=text_mask).last_hidden_state
        # H_I: (Batch, Num_Patches, d_I)
        H_I = self.vis_encoder(pixel_values=image_pixels).last_hidden_state
        
        # ----------------------------------------------------
        # Stage 1: Fact-Sentiment Feature Extraction
        # ----------------------------------------------------
        # 1. Attention Pooling to obtain semantic units (Eq 2-5)
        U_T = self.text_pooler(H_T, text_mask)  # (Batch, K_T, d_T)
        U_I = self.vis_pooler(H_I)              # (Batch, K_I, d_I)
        
        # 2. Fact space projection (Eq 6)
        f_T_F = self.MLP_T_F(U_T)  # (Batch, K_T, d_F)
        f_I_F = self.MLP_I_F(U_I)  # (Batch, K_I, d_F)
        
        # 3. Sentiment space projection (Eq 8)
        f_T_E = self.MLP_T_E(U_T)  # (Batch, K_T, d_E)
        f_I_E = self.MLP_I_E(U_I)  # (Batch, K_I, d_E)
        
        # 4. Decoupled semantic spaces construction (Eq 11-12)
        S_F_0 = torch.cat([f_T_F, f_I_F], dim=1)  # (Batch, K_T + K_I, d_F)
        S_E_0 = torch.cat([f_T_E, f_I_E], dim=1)  # (Batch, K_T + K_I, d_E)
        
        # 5. Auxiliary semantic anchoring computation (Eq 7, 10)
        loss_F = torch.tensor(0.0, device=S_F_0.device)
        loss_E = torch.tensor(0.0, device=S_E_0.device)
        
        if self.training and y_fact is not None and y_sent is not None:
            # Fact Anchoring: supervised by YOLO image labels
            f_bar_I_F = torch.mean(f_I_F, dim=1)  # Average visual fact units
            pred_fact = self.MLP_cls_F(f_bar_I_F)
            loss_F = self.bce_with_logits(pred_fact, y_fact.float())
            
            # Sentiment Anchoring: supervised by SenticNet text polarity
            f_bar_T_E = torch.mean(f_T_E, dim=1)  # Average textual sentiment units
            pred_sent = self.MLP_cls_E(f_bar_T_E)
            loss_E = self.mse_loss(pred_sent, y_sent.float())

        # ----------------------------------------------------
        # Stage 2: Feature Dynamics Evolution & Conflict-Consensus Extraction
        # ----------------------------------------------------
        # Fact Space Evolution
        S_F_prime, T_F_final = self.darfu_F(S_F_0)
        V_F = self.extractor_F(S_F_prime, T_F_final)  # (Batch, d_F)
        
        # Sentiment Space Evolution
        S_E_prime, T_E_final = self.darfu_E(S_E_0)
        V_E = self.extractor_E(S_E_prime, T_E_final)  # (Batch, d_E)

        # ----------------------------------------------------
        # Stage 3: Multi-View Deliberative Judgment
        # ----------------------------------------------------
        # V_final = concat(V^F, V^E, V^F * V^E) (Eq 27)
        V_interaction = V_F * V_E
        V_final = torch.cat([V_F, V_E, V_interaction], dim=-1)  # (Batch, 3 * feature_dim)
        
        # Final Fake News Classification (Eq 28)
        logits = self.MLP_final(V_final).squeeze(-1)
        prob = torch.sigmoid(logits)
        
        return prob, loss_F, loss_E

# =====================================================================
# 5. Trainer Class (Preserving your original framework layout)
# =====================================================================
class DOMAINTrainerWeibo():
    def __init__(self, lr, weight_decay, train_loader, val_loader, test_loader, 
                 save_param_dir, use_cuda=True, epoches=50, lambda_F=0.075, lambda_E=0.075):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epoches = epoches
        self.use_cuda = use_cuda
        self.save_param_dir = save_param_dir
        
        # Model hyperparameters based on Section 4 of the paper
        self.lambda_F = lambda_F
        self.lambda_E = lambda_E
        
        if not os.path.exists(self.save_param_dir):
            os.makedirs(self.save_param_dir, exist_ok=True)

        # Initialize DCCF Model instead of DIVER
        self.model = DCCFModel(M=4, tau=1.5, K_T=10, K_I=10)
        if self.use_cuda: 
            self.model = self.model.cuda()

        # Optimizer and Main Loss (Eq 29)
        self.bce_loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model_save_filename = 'dccf_best_model.pkl'

    def clipdata2gpu(self, batch):
        """Helper to move batch dictionary to GPU"""
        if not self.use_cuda: return batch
        return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.epoches):
            self.model.train()
            train_iter = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epoches}")
            total_loss = 0.0

            for batch in train_iter:
                batch_data = self.clipdata2gpu(batch)
                
                # Assume dataloader provides 'label' for fake news
                labels = batch_data.get('label').float()
                
                # Forward pass - returns prob, and the two auxiliary anchoring losses
                final_prob, loss_F, loss_E = self.model(**batch_data)
                
                # Eq 29: Final Fake News Prediction BCE Loss
                loss_final = self.bce_loss(final_prob, labels)
                
                # Eq 30: Total objective combining final loss + weakly anchored spaces
                loss = loss_final + (self.lambda_F * loss_F) + (self.lambda_E * loss_E)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                train_iter.set_postfix(
                    total_loss=loss.item(), 
                    l_final=loss_final.item(), 
                    l_F=loss_F.item(), 
                    l_E=loss_E.item()
                )

            avg_train_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}')
            
            # Validation Step
            val_loss = self.test(self.val_loader, mode='val')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Validation improved. Saving model...")
                torch.save(self.model.state_dict(), os.path.join(self.save_param_dir, self.model_save_filename))

        # Final Test evaluation
        print("Training complete. Running Final Test...")
        best_path = os.path.join(self.save_param_dir, self.model_save_filename)
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location='cpu'))
        self.test(self.test_loader, mode='test')

    def test(self, dataloader, mode='test'):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=f"Running {mode}"):
                batch_data = self.clipdata2gpu(batch)
                labels = batch_data.get('label').float()
                
                # Forward pass without calculating aux losses (y_fact/y_sent not strictly required for inference)
                final_prob, loss_F, loss_E = self.model(**batch_data)
                
                loss_final = self.bce_loss(final_prob, labels)
                
                # Optional: We calculate total loss exactly like training if labels are present
                loss = loss_final + (self.lambda_F * loss_F) + (self.lambda_E * loss_E)
                total_loss += loss.item()
                
                # Convert probabilities to binary predictions
                preds = (final_prob > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        
        if mode == 'test':
            # Simplified metric calculation representing Table 1 Accuracy in paper
            correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
            acc = correct / len(all_labels) if len(all_labels) > 0 else 0
            print(f"Test Accuracy: {acc * 100:.2f}%")
            
        return avg_loss

# =====================================================================
# 6. Dummy Dataloader & Execution Wrapper (To ensure code completeness)
# =====================================================================
class DummyWeiboDataset(Dataset):
    """
    A simulated dataset wrapper constructed purely to demonstrate how the DCCFModel 
    expects inputs, allowing this complete file to run out of the box.
    """
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Fake Text: Bert sequence of length 197 as defined in Section 4.1
        text_ids = torch.randint(0, 21128, (197,))
        text_mask = torch.ones(197)
        
        # Fake Image: ViT input of 3x224x224
        image_pixels = torch.randn(3, 224, 224)
        
        # Label: 1 for Fake News, 0 for Real News
        label = torch.randint(0, 2, (1,)).squeeze(0)
        
        # YOLO Multi-hot pseudo-labels for Fact Space Anchoring (e.g., 80 COCO classes)
        y_fact = torch.randint(0, 2, (80,)).float()
        
        # SenticNet polarity score for Sentiment Space Anchoring (e.g., [-1.0, 1.0])
        y_sent = torch.randn(1) * 0.5
        
        return {
            'text_ids': text_ids,
            'text_mask': text_mask,
            'image_pixels': image_pixels,
            'y_fact': y_fact,
            'y_sent': y_sent,
            'label': label
        }

if __name__ == "__main__":
    print("Initializing dummy dataset to verify framework integrity...")
    
    # Initialize datasets
    train_ds = DummyWeiboDataset(size=128)
    val_ds   = DummyWeiboDataset(size=32)
    test_ds  = DummyWeiboDataset(size=32)
    
    # Wrap in dataloaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    # Define save directory
    save_dir = "./dccf_checkpoints"
    
    # Launch trainer ensuring original framework handles the new DCCF architecture
    trainer = DOMAINTrainerWeibo(
        lr=1e-4, 
        weight_decay=1e-5, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        save_param_dir=save_dir, 
        use_cuda=torch.cuda.is_available(), 
        epoches=2,            # Short epoch for testing logic
        lambda_F=0.075,       # Hyperparameter from Sec 4.1
        lambda_E=0.075        # Hyperparameter from Sec 4.1
    )
    
    print("Starting Training Loop with DCCF implementation...")
    trainer.train()