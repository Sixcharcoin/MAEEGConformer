import torch
import torch.nn as nn
from typing import Tuple

class PositionalEncodingV2(nn.Module):
    """
    Transformer/Conformer向けの位置エンコーディング層。
    時系列データにおいて「どの時点のデータか」という順番情報をモデルに与える。
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        # 固定のサイン・コサイン波ではなく、学習によって最適な位置情報を獲得するパラメータ
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (Batch, Time, Dim) """
        return x + self.pos_embed[:, :x.size(1), :]

class Swish(nn.Module):
    """ Swish活性化関数。ReLUよりも滑らかな勾配を持ち、深いネットワークで有効。 """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class GatedAttentionPooling(nn.Module):
    """
    時系列の特徴量を1つのベクトルに集約するためのプーリング層。
    単純な平均(Average Pooling)とは異なり、重要な時間に高い重みを付ける（Attention）。
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear_v = nn.Linear(d_model, d_model) # 特徴変換 (Tanh用)
        self.linear_u = nn.Linear(d_model, d_model) # ゲート開閉 (Sigmoid用)
        self.linear_w = nn.Linear(d_model, 1)       # 重み(スコア)計算用

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, SeqLen, d_model)
        v = torch.tanh(self.linear_v(x))
        u = torch.sigmoid(self.linear_u(x))
        gated_features = v * u  # 情報を通す量(ゲート)を制御
        
        scores = self.linear_w(gated_features) # (B, SeqLen, 1)
        weights = torch.softmax(scores, dim=1)
        
        # 重みに従って時間軸(SeqLen)方向に加重平均をとる
        output = (x * weights).sum(dim=1) # (B, d_model)
        return output

class ConformerBlockV2(nn.Module):
    """
    Conformerの基本ブロック (Macaron-like 構造)。
    [FFN (1/2) -> Self-Attention -> Convolution -> FFN (1/2)] の順で処理を行う。
    これにより、Transformerの「大域的な文脈理解」とCNNの「局所的な特徴抽出」を両立する。
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, conv_kernel_size: int = 31, dropout: int = 0.1):
        super().__init__()
        
        # 1. Feed Forward Module (前半の半分)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # 2. Multi-Head Self-Attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        # 3. Convolution Module (時系列の局所パターンの抽出)
        self.conv_norm = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2, groups=d_model, # Depth-wise Conv
        )
        self.dw_norm = nn.LayerNorm(d_model)
        self.dw_act = Swish()
        self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_dropout = nn.Dropout(dropout)

        # 4. Feed Forward Module (後半の半分)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Dim)
        x = x + 0.5 * self.ffn1(x)
        
        x_norm = self.attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.attn_dropout(attn_out)

        residual = x
        x_norm = self.conv_norm(x).transpose(1, 2) # Conv1d用にするため (B, D, T)
        x_conv = self.glu(self.pw_conv1(x_norm))
        x_conv = self.dw_conv(x_conv).transpose(1, 2)
        x_conv = self.dw_act(self.dw_norm(x_conv)).transpose(1, 2)
        x_conv = self.pw_conv2(x_conv).transpose(1, 2)
        x = residual + self.conv_dropout(x_conv)

        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)
