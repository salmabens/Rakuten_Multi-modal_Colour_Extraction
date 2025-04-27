import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, classification_report
from multiprocessing import freeze_support
from PIL import Image

def main():
    OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), '../Output') 
    PROCESSED_TRAIN = os.path.join(OUTPUT_DIR, "processed_images_train")
    BATCH_SIZE      = 64
    LR              = 1e-4
    EPOCHS          = 20
    VAL_SPLIT       = 0.2
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience        = 3

    # === 1) Chargement du DataFrame & des features/text/labels pré-extraits ===
    df_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "X_train_clean.csv"))
    df_clean["filename"] = (
        "image_" + df_clean.imageid.astype(str)
        + "_product_" + df_clean.productid.astype(str)
        + ".jpg"
    )

    # Features image & labels : extraits uniquement sur les images existantes
    X_img_all  = torch.load(os.path.join(OUTPUT_DIR, "effB0_features.pt"))   # [N,1280]
    labels_all = torch.load(os.path.join(OUTPUT_DIR, "train_labels.pt"))     # [N]

    # Features texte pour TOUTES les lignes
    X_txt_all  = np.load(os.path.join(OUTPUT_DIR, "X_w2v_train.npy"))        # [M,300]
  
    present = set(os.listdir(PROCESSED_TRAIN))
    mask    = df_clean["filename"].isin(present).values
    idx     = np.where(mask)[0]

    X_img  = X_img_all                               # (N,1280)
    labels = labels_all                              # (N,)
    X_txt  = torch.from_numpy(X_txt_all[idx]).float()  # (N,300)

    assert X_img.size(0) == X_txt.size(0) == labels.size(0), \
        f"Mismatch samples: img {X_img.size(0)}, txt {X_txt.size(0)}, lab {labels.size(0)}"

    num_classes = int(labels.max().item()) + 1

    # === 2) Dataset multimodal ===
    class MultimodalDataset(Dataset):
        def __init__(self, img_feats, txt_feats, y):
            self.img = img_feats
            self.txt = txt_feats
            self.y   = y
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            return (self.img[i], self.txt[i]), self.y[i]

    ds     = MultimodalDataset(X_img, X_txt, labels)
    n_val  = int(len(ds) * VAL_SPLIT)
    n_tr   = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val],
                                generator=torch.Generator().manual_seed(42))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=True)

    # === 3) Modèle de fusion ===
    class FusionMLP(nn.Module):
        def __init__(self, dim_img, dim_txt, numc):
            super().__init__()
            D = dim_img + dim_txt
            self.net = nn.Sequential(
                nn.Linear(D, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(1024,  512), nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512,    numc)
            )
        def forward(self, x):
            img, txt = x
            return self.net(torch.cat([img, txt], dim=1))

    model = FusionMLP(X_img.size(1), X_txt.size(1), num_classes).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    crit  = nn.CrossEntropyLoss()

    # === 4) Entraînement & early stopping ===
    best_f1, wait = 0.0, 0
    for ep in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        for (im, tx), yb in tr_ld:
            im, tx, yb = im.to(DEVICE), tx.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model((im, tx))
            loss   = crit(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * yb.size(0)
        train_loss /= n_tr

        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for (im, tx), yb in va_ld:
                im, tx = im.to(DEVICE), tx.to(DEVICE)
                logits = model((im, tx))
                all_p.append(logits.argmax(1).cpu().numpy())
                all_t.append(yb.numpy())
        all_p = np.concatenate(all_p)
        all_t = np.concatenate(all_t)
        f1    = f1_score(all_t, all_p, average="weighted")

        print(f"Epoch {ep}/{EPOCHS} — TrainLoss: {train_loss:.4f} — Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1, wait = f1, 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_fusion.pth"))
        else:
            wait += 1
            if wait >= patience:
                print("⇢ Early stopping")
                break

    print("\n=== Rapport final ===")
    print(classification_report(all_t, all_p, digits=4))
    print(f" Best weighted F1 = {best_f1:.4f} (modèle sauvé dans {OUTPUT_DIR})")

if __name__ == "__main__":
    freeze_support()
    main()
