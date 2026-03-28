import torch
import torch.nn as nn
import copy
from typing import Tuple
from .layers import PositionalEncodingV2, ConformerBlockV2, GatedAttentionPooling #refer to model_components.py

class ConformerMAEEG_Pretrain(nn.Module):
    """
    EEGデータのためのMasked Autoencoder (MAE) モデル。
    入力波形の一部をランダムに隠し（マスクし）、残りの部分から隠された部分を予測・復元させることで、
    ラベルデータなしにEEGの構造的特徴（表現）を学習する。
    """
    def __init__(self, in_channels=17, time_steps=100, embed_dim=128, mask_ratio=0.6, depth=4, num_subjects=11):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # パッチ埋め込み: 生のEEG信号(チャンネル数)を高次元の特徴空間にマッピングする
        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        
        self.pos_encoder = PositionalEncodingV2(embed_dim, max_len=time_steps)
        
        # 被験者ごとの脳波の個人差を吸収するためのEmbedding
        self.subject_embed = nn.Embedding(num_subjects, embed_dim)
        nn.init.xavier_uniform_(self.subject_embed.weight)

        # Encoder: Conformerを使用して局所・大域特徴を抽出
        self.encoder_blocks = nn.ModuleList([
            ConformerBlockV2(d_model=embed_dim, n_heads=4, conv_kernel_size=15, dropout=0.1)
            for _ in range(depth)
        ])
        
        # Decoder: 軽量なTransformer（復元タスクは表現学習の補助であるため浅くてよい）
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)

        # マスクトークン: 隠された部分をDecoderで表現するための学習可能なダミーベクトル
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.output_head = nn.Linear(embed_dim, in_channels)

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        系列の一部をランダムに取り除く関数。
        argsortを2回適用することで「シャッフルされたインデックス」と「元に戻すためのインデックス」
        を同時に取得する、MAE特有のスマートな実装。
        """
        B, T, D = x.shape
        len_keep = int(T * (1.0 - self.mask_ratio)) # 保持する系列長

        noise = torch.rand(B, T, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)        # シャッフル順
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 元に戻す順序
        
        # 保持するインデックスだけを取得
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 損失計算用に、どこをマスクしたかを示すバイナリマップを作成（1:マスクされた, 0:保持された）
        mask = torch.ones(B, T, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor, subject_idxs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. パッチ化と埋め込み
        feat = self.patch_embed(x).transpose(1, 2)  # (B, T, D)
        feat = feat + self.pos_encoder.pos_embed[:, :feat.size(1), :]
        feat = feat + self.subject_embed(subject_idxs).unsqueeze(1)

        # 2. マスキング (情報を捨てる)
        latent, mask, ids_restore = self.random_masking(feat)

        # 3. Encoder (保持された少数のパッチのみを処理するため高速)
        for block in self.encoder_blocks:
            latent = block(latent)

        # 4. Decoderに向けた復元準備 (マスクトークンの挿入)
        B, T_vis, D = latent.shape
        T_full = feat.size(1)
        mask_tokens = self.mask_token.expand(B, T_full - T_vis, D)
        
        # 保持された特徴とマスクトークンを結合し、元の順序に並べ直す
        x_ = torch.cat([latent, mask_tokens], dim=1)
        x_rec = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x_rec = x_rec + self.pos_encoder.pos_embed[:, :T_full, :]

        # 5. Decoderと出力 (元の波形を予測)
        decoded = self.decoder(x_rec)
        pred = self.output_head(decoded).transpose(1, 2) # (B, Channels, Time)

        return pred, mask


class ConformerMAEEG_Classifier(nn.Module):
    """
    事前学習されたMAEEGのEncoderを利用し、下流タスク（分類）を行うモデル。
    """
    def __init__(self, pretrained_model: ConformerMAEEG_Pretrain, num_classes: int = 5):
        super().__init__()
        
        # 事前学習済みモデルからEncoder部分の重みをコピー (deepcopyで独立させる)
        self.patch_embed  = copy.deepcopy(pretrained_model.patch_embed)
        self.pos_encoder  = copy.deepcopy(pretrained_model.pos_encoder)
        self.encoder_blocks = copy.deepcopy(pretrained_model.encoder_blocks)
        self.subject_embed  = copy.deepcopy(pretrained_model.subject_embed)

        embed_dim = pretrained_model.output_head.in_features
        
        # 分類用のヘッドを追加
        self.pool = GatedAttentionPooling(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(x).transpose(1, 2)
        feat = feat + self.pos_encoder.pos_embed[:, :feat.size(1), :]
        feat = feat + self.subject_embed(subject_idxs).unsqueeze(1)

        # 分類時はマスキングせず、すべての系列をEncoderに通す
        for block in self.encoder_blocks:
            feat = block(feat)

        # プーリング層で時間軸を潰して1つのベクトルにし、分類器へ渡す
        feat_pooled = self.pool(feat)
        out = self.classifier(feat_pooled)
        return out
