import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataset import ThingsEEGDataset
from models.maeeg import ConformerMAEEG_Pretrain, ConformerMAEEG_Classifier

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.8):
    """
    Mixup: 2つのサンプルとそのラベルを一定の比率(lam)で混ぜ合わせるデータ拡張手法。
    決定境界を滑らかにし、未知のデータに対する過学習を防ぐ。
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """ Mixupされたラベルに対する損失を計算する。 """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 事前学習モデルの定義と重みのロード
    pretrain_model = ConformerMAEEG_Pretrain(
        in_channels=17, time_steps=100, embed_dim=128, depth=4, num_subjects=11
    ).to(device)

    try:
        pretrain_model.load_state_dict(torch.load("mae_pretrained.pt", map_location=device))
        print("Pre-trained weights loaded successfully.")
    except FileNotFoundError:
        print("Warning: 'mae_pretrained.pt' not found. Starting with random initialization.")

    # 2. 分類モデルの構築
    model = ConformerMAEEG_Classifier(pretrain_model, num_classes=5).to(device)

    # 3. データローダーの準備
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        ThingsEEGDataset("train"), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        ThingsEEGDataset("val"), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    max_val_acc = 0.0

    # -------------------------------------------------------------
    # Phase 1: Linear Probing (Headのみの学習)
    # 事前学習で獲得した強力な表現（重み）が、ランダム初期化された
    # 分類ヘッドからの大きな勾配によって破壊される(Catastrophic Forgetting)のを防ぐ。
    # -------------------------------------------------------------
    print("--- Phase 1: Linear Probing ---")
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    for p in model.pool.parameters(): p.requires_grad = True

    optimizer_warmup = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(5):
        model.train()
        for X, y, subj_idx in train_loader:
            X, y, subj_idx = X.to(device), y.to(device), subj_idx.to(device)
            optimizer_warmup.zero_grad()
            loss = criterion(model(X, subj_idx), y)
            loss.backward()
            optimizer_warmup.step()
        print(f"Warmup Epoch {epoch+1}/5 completed.")

    # -------------------------------------------------------------
    # Phase 2: Full Fine-Tuning (モデル全体の微調整)
    # -------------------------------------------------------------
    print("--- Phase 2: Full Fine-tuning ---")
    for p in model.parameters(): p.requires_grad = True

    epochs = 35
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = [], []
        
        for X, y, subj_idx in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            X, y, subj_idx = X.to(device), y.to(device), subj_idx.to(device)
            
            # Mixupの適用
            mixed_x, y_a, y_b, lam = mixup_data(X, y, alpha=0.8)
            preds = model(mixed_x, subj_idx)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # Mixup時は正確なAccuracy計算が難しいため、代表ラベルとの一致を見る
            train_acc.append((preds.argmax(dim=-1) == y).float().mean().item())

        # Validation
        model.eval()
        val_loss, val_acc = [], []
        with torch.no_grad():
            for X, y, subj_idx in val_loader:
                X, y, subj_idx = X.to(device), y.to(device), subj_idx.to(device)
                preds = model(X, subj_idx)
                val_loss.append(criterion(preds, y).item())
                val_acc.append((preds.argmax(dim=-1) == y).float().mean().item())

        avg_val_acc = np.mean(val_acc)
        print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_loss):.4f} | Val Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_acc)

        if avg_val_acc > max_val_acc:
            print(f"New Best! {max_val_acc:.4f} -> {avg_val_acc:.4f}. Saving model...")
            torch.save(model.state_dict(), "mae_classifier_best.pt")
            max_val_acc = avg_val_acc

if __name__ == "__main__":
    main()
