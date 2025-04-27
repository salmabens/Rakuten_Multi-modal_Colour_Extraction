import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), '../Output') 
    PROCESSED_TRAIN = os.path.join(OUTPUT_DIR, "processed_images_train")
    BATCH_SIZE      = 64
    LR              = 1e-4
    EPOCHS          = 10
    VAL_SPLIT       = 0.2
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "X_train_clean.csv"))
    df_clean["filename"] = (
        "image_" + df_clean.imageid.astype(str)
        + "_product_" + df_clean.productid.astype(str)
        + ".jpg"
    )
    all_filenames = df_clean["filename"].values

    # Lister les images réellement présentes et calculer orig_idx
    present = set(os.listdir(PROCESSED_TRAIN))
    mask = np.array([fn in present for fn in all_filenames])
    orig_idx = np.where(mask)[0]

    # Charger features et labels
    X_img_all  = torch.load(os.path.join(OUTPUT_DIR, "effB0_features.pt"))
    labels_all = torch.load(os.path.join(OUTPUT_DIR, "train_labels.pt"))
    X_txt_all  = np.load(os.path.join(OUTPUT_DIR, "X_w2v_train.npy"))

    # Conserver X_img_all et labels_all tels quels, ne filtrer que le texte
    X_img = X_img_all
    labels = labels_all
    X_txt  = torch.from_numpy(X_txt_all[orig_idx]).float()

    assert X_img.size(0) == X_txt.size(0) == labels.size(0), \
        f"Mismatch: {X_img.size(0)}, {X_txt.size(0)}, {labels.size(0)}"

    # Concaténation multimodale
    X = torch.cat([X_img, X_txt], dim=1)
    num_classes = int(labels.max().item() + 1)

    # Dataset & split train/val
    class FusedDataset(Dataset):
        def __init__(self, X, y):
            self.X = X; self.y = y
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]

    ds = FusedDataset(X, labels)
    n_val = int(len(ds) * VAL_SPLIT)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Modèle MLP fusionné
    class FusionMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    model = FusionMLP(X.size(1), num_classes).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()

    # Entraînement & évaluation
    for ep in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= n_train

        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                all_preds.append(out.argmax(1).cpu().numpy())
                all_trues.append(yb.numpy())
        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)
        f1 = f1_score(all_trues, all_preds, average="weighted")
        print(f"Epoch {ep}/{EPOCHS} — Train loss: {train_loss:.4f} — Val F1 (weighted): {f1:.4f}")

    # Rapport complet
    print("\nClassification report (validation):")
    print(classification_report(all_trues, all_preds, digits=4))

    # Matrice de confusion 
    cm = confusion_matrix(all_trues, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(12,10))
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=True,
        fmt="d",
        annot_kws={"size":6},
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        cbar_kws={"shrink": .75}
    )
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité terrain")
    plt.title("Matrice de confusion (validation)")
    out_cm = os.path.join(OUTPUT_DIR, "confusion_matrix_validation.png")
    plt.savefig(out_cm, bbox_inches="tight", dpi=150)
    print(f" Matrice de confusion sauvegardée dans : {out_cm}")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "fusion_mlp.pth"))
    print(f"\n Modèle fusion sauvegardé dans {OUTPUT_DIR}")

if __name__ == "__main__":
    freeze_support()
    main()
