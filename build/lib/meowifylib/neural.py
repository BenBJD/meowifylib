import math

import lightning as L
import torch
import torch.nn as nn

# import torchtune.modules
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
        loss = (
            self.alpha * (1 - p) ** self.gamma * ce_loss
            + (1 - self.alpha) * p**self.gamma * ce_loss
        )
        return loss.mean()


# Adapted from https://web.archive.org/web/20230315052215/https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Args:
        - x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
        - Tensor with positional encoding added
        """
        seq_len = x.size(1)
        device = x.device

        position = torch.arange(0, seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device)
            * -(math.log(10000.0) / self.embed_dim)
        )  # (embed_dim/2)

        pe = torch.zeros(seq_len, self.embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, seq_len, embed_dim)
        return x + pe


class MeowifyVocal2MIDINet(L.LightningModule):
    def __init__(self, lr=0.0001, warmup_epochs=5, epochs=20):
        super().__init__()

        self.smoothing = 0.0  # unused
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs

        self.conv1d_1 = nn.Conv1d(
            in_channels=120, out_channels=512, kernel_size=7, padding=3
        )
        self.bn1d_1 = nn.BatchNorm1d(512)

        self.conv1d_2 = nn.Conv1d(
            in_channels=512, out_channels=256, kernel_size=5, padding=2
        )
        self.bn1d_2 = nn.BatchNorm1d(256)

        self.conv1d_4 = nn.Conv1d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.bn1d_4 = nn.BatchNorm1d(512)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embed_dim=512)

        # Attention Layers
        self.attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=4, batch_first=True
        )
        self.attention_norm1 = nn.LayerNorm(512)
        self.attention_2 = nn.MultiheadAttention(
            embed_dim=512, num_heads=4, batch_first=True
        )
        self.attention_norm2 = nn.LayerNorm(512)

        # Output Layer
        self.linear_output = nn.Linear(in_features=512, out_features=60)

        # Activations/Utilities
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Metrics
        self.total_train_loss = 0.0
        self.total_val_loss = 0.0
        self.training_losses = []
        self.val_losses = []
        self.train_precision = []  # Unused
        self.train_recall = []  # Unused
        self.train_f1 = []  # Unused
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        self.train_auc = []  # Unused
        self.val_auc = []
        self.train_map = []  # Unused
        self.val_map = []
        self.learning_rates = []

    def forward(self, x):
        """
        x: input tensor with shape (batch, 120 notes, frames)
        Returns:
        output tensor with shape (batch, 60 notes, frames)
        """
        # Downsample notes
        x = self.relu(self.bn1d_1(self.conv1d_1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn1d_2(self.conv1d_2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn1d_4(self.conv1d_4(x)))
        x = self.dropout(x)

        # Permute for Attention
        x = x.permute(0, 2, 1)  # (batch, frames, features)

        # Positional Encoding
        x = self.positional_encoding(x)

        # Attention Layers
        x, _ = self.attention(x, x, x)
        x = self.attention_norm1(x)  # Normalize attention output
        x = self.dropout(x)  # Dropout after attention

        x, _ = self.attention_2(x, x, x)
        x = self.attention_norm2(x)  # Normalize second attention output
        x = self.dropout(x)  # Dropout after second attention

        # Final Linear Output
        x = self.linear_output(x)
        x = x.permute(0, 2, 1)  # (batch, 60 notes, 1024 frames)

        return x

    def training_step(self, batch, batch_idx):
        stft, target_midi = batch
        output = self.forward(stft)

        target_midi = (1 - self.smoothing) * target_midi  # Not smoothing used

        focal_fn = FocalLoss(alpha=0.75, gamma=2)
        focal_loss = focal_fn(output, target_midi)

        # Combine losses
        total_loss = focal_loss

        if batch_idx == 0:
            self.training_losses.append(
                self.total_train_loss / self.trainer.num_training_batches
            )
            self.total_train_loss = 0.0
            # Track learning rate
            current_lr = self.lr_schedulers().get_last_lr()[-1]
            self.learning_rates.append(current_lr)

        self.total_train_loss += float(total_loss)

        # Logging for monitoring
        self.log(
            "train_loss",
            self.total_train_loss / self.trainer.num_training_batches,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        current_lr = self.lr_schedulers().get_last_lr()[-1]
        self.log("lr", current_lr, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        stft, target_midi = batch
        output = self.forward(stft)

        focal_fn = FocalLoss(alpha=0.75, gamma=2)
        focal_loss = focal_fn(output, target_midi)

        # Combine losses
        total_loss = focal_loss

        if batch_idx == 0:
            self.val_losses.append(
                self.total_val_loss / self.trainer.num_val_batches[0]
            )
            self.total_val_loss = 0.0

            # Calculate additional metrics at the start of each epoch
            with torch.no_grad():
                # Convert logits to probabilities
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).float()

                # Flatten tensors for metric calculation
                preds_flat = preds.detach().cpu().numpy().flatten()
                target_flat = target_midi.detach().cpu().numpy().flatten()
                probs_flat = probs.detach().cpu().numpy().flatten()

                # Calculate precision, recall, and F1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    target_flat, preds_flat, average="binary", zero_division=0
                )
                self.val_precision.append(precision)
                self.val_recall.append(recall)
                self.val_f1.append(f1)

                # Calculate AUC-ROC if possible
                try:
                    auc = roc_auc_score(target_flat, probs_flat)
                    self.val_auc.append(auc)
                except ValueError:
                    # This can happen if there's only one class in the batch
                    self.val_auc.append(0.5)  # Default value for random classifier

                # Calculate mean average precision
                try:
                    map_score = average_precision_score(target_flat, probs_flat)
                    self.val_map.append(map_score)
                except ValueError:
                    self.val_map.append(0.0)

        self.total_val_loss += float(total_loss)

        # Calculate metrics for each batch for real-time feedback
        with torch.no_grad():
            # Convert logits to probabilities
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            # Flatten tensors for metric calculation
            preds_flat = preds.detach().cpu().numpy().flatten()
            target_flat = target_midi.detach().cpu().numpy().flatten()
            probs_flat = probs.detach().cpu().numpy().flatten()

            # Calculate precision, recall, and F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_flat, preds_flat, average="binary", zero_division=0
            )

        # Logging for monitoring
        self.log(
            "val_loss",
            self.total_val_loss / self.trainer.num_val_batches[0],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_precision", precision, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log("val_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch) / float(max(1, self.warmup_epochs))
            return max(
                0.01,
                0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (epoch - self.warmup_epochs)
                        / (self.epochs - self.warmup_epochs)
                    )
                ),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list):
            outputs = self.forward(batch[0])
            return torch.sigmoid(outputs), batch[1]
        outputs = self.forward(batch)
        return torch.sigmoid(outputs)
