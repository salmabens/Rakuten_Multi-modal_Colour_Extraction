import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report
from multiprocessing import freeze_support

# -----------------------------------------------------------------------------
# 1) Classes & fonctions au niveau module
# -----------------------------------------------------------------------------
class FusionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class FusionMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        p_t = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean()

# -----------------------------------------------------------------------------
# 2) Main
# -----------------------------------------------------------------------------
def main():
    # ——— Réglages chemins & hyperparamètres ———
    OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), '../Output') 
    CSV_CLEAN     = os.path.join(OUTPUT_DIR, "X_train_clean.csv")
    IMG_FEAT_PATH = os.path.join(OUTPUT_DIR, "effB0_features.pt")
    TXT_FEAT_PATH = os.path.join(OUTPUT_DIR, "X_w2v_train.npy")
    LABELS_PATH   = os.path.join(OUTPUT_DIR, "train_labels.pt")
    PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed_images_train")

    BATCH_SIZE = 128
    LR         = 1e-4
    EPOCHS     = 20
    VAL_SPLIT  = 0.2
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CSV nettoyé 
    df = pd.read_csv(CSV_CLEAN)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
    df["filename"] = (
        "image_" + df.imageid.astype(str)
      + "_product_" + df.productid.astype(str)
      + ".jpg"
    )

    # masque des images présentes
    present = set(os.listdir(PROCESSED_DIR))
    mask    = df["filename"].isin(present).values
    idx     = np.where(mask)[0]

    # Chargement des embeddings et labels
    X_img_all = torch.load(IMG_FEAT_PATH)         # (N_valid,1280)
    X_txt_all = np.load(TXT_FEAT_PATH)            # (N_orig,300)
    y_all     = torch.load(LABELS_PATH).numpy()   # (N_valid,)

    # alignement image ↔ texte ↔ labels
    X_txt = torch.from_numpy(X_txt_all[idx]).float()  
    y     = torch.from_numpy(y_all).long()           
    assert X_img_all.size(0) == X_txt.size(0) == y.size(0)

    # Concaténation multimodale 
    X_all       = torch.cat([X_img_all, X_txt], dim=1)  # (N_valid,1580)
    num_classes = int(y.max().item()) + 1

    # Poids de classes pour la loss 
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y.numpy()
    )
    class_weights = torch.tensor(class_weights, device=DEVICE, dtype=torch.float)

    # Dataset & split train/val 
    ds = FusionDataset(X_all, y)
    n_val = int(len(ds) * VAL_SPLIT)
    n_tr  = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Modèle, optim, loss 
    model     = FusionMLP(X_all.size(1), num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss(gamma=2.0, weight=class_weights)

    # Entraînement & validation 
    best_f1 = 0.0
    for ep in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
        train_loss = running_loss / n_tr

        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE)
                logits = model(Xb)
                preds  = logits.argmax(1).cpu().numpy()
                all_preds.append(preds)
                all_trues.append(yb.numpy())
        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)
        f1 = f1_score(all_trues, all_preds, average="weighted")

        print(f"[{ep}/{EPOCHS}] train_loss={train_loss:.4f}  val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, "fusion_mlp_focal.pth")
            )

    # Rapport final 
    print("\n=== Classification Report (val) ===")
    print(classification_report(all_trues, all_preds, digits=4))
    print(f"✅ Meilleur F1 pondéré = {best_f1:.4f} (fusion_mlp_focal.pth)")

if __name__ == "__main__":
    freeze_support()
    main()
