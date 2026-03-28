import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional

class ThingsEEGDataset(Dataset):
    """
    ThingsEEGデータセットを読み込み、PyTorchで扱える形式に変換するクラス。
    
    訓練時には過学習を防ぐため、確率的なデータ拡張（Data Augmentation）を適用します。
    """
    def __init__(self, split: str, data_dir: str = "data/") -> None:
        """
        Args:
            split (str): "train", "val", "test" のいずれか。
            data_dir (str): データが保存されているディレクトリのパス。
        """
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 5
        self.num_subjects = 10  # Note: 元コードでは10と11が混在していましたが、統一設定として扱うのがベターです。

        # 脳波データと被験者IDの読み込み
        self.X = torch.from_numpy(np.load(f"{data_dir}/{split}/eeg.npy")).to(torch.float32)
        self.subject_idxs = torch.from_numpy(np.load(f"{data_dir}/{split}/subject_idxs.npy"))

        # train, val の場合は正解ラベルも読み込む
        if split in ["train", "val"]:
            self.y = torch.from_numpy(np.load(f"{data_dir}/{split}/labels.npy"))

        print(f"[{split.upper()}] EEG: {self.X.shape}, Subject indices: {self.subject_idxs.shape}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        x = self.X[i].clone()  # 元データを書き換えないようにクローンを作成

        # 1. 正規化 (Standardization)
        # サンプル（波形）単位で平均0、分散1になるように正規化し、スケールを揃える。
        mean = x.mean()
        std = x.std()
        x = (x - mean) / (std + 1e-6)  # ゼロ除算防止のための微小値(1e-6)

        # 2. データ拡張 (Data Augmentation) - 訓練時のみ
        # 全ての拡張をかけると波形が破壊されるため、p_augの確率で1つだけ適用する
        if self.split == 'train':
            p_aug = 0.7
            if torch.rand(1).item() < p_aug:
                # 0~4の乱数を生成し、適用する拡張手法を決定
                aug_type = torch.randint(0, 5, (1,)).item()
                
                if aug_type == 0:
                    # Noise: 微小なガウシアンノイズを付与して堅牢性を上げる
                    noise = torch.randn_like(x) * 0.05
                    x = x + noise
                elif aug_type == 1:
                    # Shift: 波形を時間軸方向に少しずらす（位相の変化に強くする）
                    shift = torch.randint(-3, 3, (1,)).item()
                    x = torch.roll(x, shifts=shift, dims=1)
                elif aug_type == 2:
                    # Channel Dropout: ランダムなチャンネル（電極）をゼロにして欠損への耐性をつける
                    num_drop = torch.randint(1, 4, (1,)).item()
                    drop_idx = torch.randperm(x.shape[0])[:num_drop]
                    x[drop_idx] = 0
                elif aug_type == 3:
                    # Time Mask: 特定の時間帯のデータをゼロにする（一時的なノイズや瞬きの影響を模倣）
                    t_start = torch.randint(0, 85, (1,)).item()
                    t_len = torch.randint(5, 15, (1,)).item()
                    x[:, t_start:t_start+t_len] = 0
                elif aug_type == 4:
                    # Scaling: 振幅をランダムに増減させる
                    scale = torch.empty(1).uniform_(0.8, 1.2).item()
                    x = x * scale

        # ラベルが存在する場合（train, val）としない場合（test）で戻り値を変える
        if hasattr(self, "y"):
            return x, self.y[i], self.subject_idxs[i]
        else:
            return x, self.subject_idxs[i]
