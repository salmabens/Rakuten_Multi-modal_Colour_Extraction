import os
from multiprocessing import freeze_support
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import timm  # on passe par timm pour Swin

DATA_PATH = os.path.join(os.path.dirname(__file__), '../Data') 
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../Output')  
PROCESSED_TRAIN = os.path.join(OUTPUT_DIR, "processed_images_train")

BATCH_IMG = 32
LR        = 1e-4
EPOCHS    = 5
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1) Transforms & Datasets
# =============================================================================
transform_preserve = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

class ImageDataset(Dataset):
    def __init__(self, folder, df, transform=None):
        self.transform = transform
        present = set(os.listdir(folder))
        self.samples = []
        for _, row in df.iterrows():
            fname = f"image_{row.imageid}_product_{row.productid}.jpg"
            if fname in present:
                self.samples.append((os.path.join(folder, fname), row.prdtypecode))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# =============================================================================
# 2) Main
# =============================================================================
def main():
    # 2.1) Chargement + concaténation
    X = pd.read_csv(os.path.join(DATA_DIR, "X_train_update.csv"))\
          .drop(columns="Unnamed: 0")
    y = pd.read_csv(os.path.join(DATA_DIR, "Y_train_CVw08PX.csv"))\
          .drop(columns="Unnamed: 0")
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    # 2.2) Encodage
    le = LabelEncoder()
    df["prdtypecode"] = le.fit_transform(df["prdtypecode"])
    num_classes = df["prdtypecode"].nunique()

    # 2.3) DataLoader images  
    ds_img = ImageDataset(PROCESSED_TRAIN, df, transform_preserve)
    loader = DataLoader(ds_img, batch_size=BATCH_IMG,
                        shuffle=False, num_workers=4, pin_memory=True)

    # =============================================================================
    # 3) Extraction des features
    # =============================================================================
    # 3.1) EfficientNet-B0 (torchvision)
    model_eff = models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.DEFAULT
                ).to(DEVICE).eval()
    for p in model_eff.parameters(): p.requires_grad = False

    # 3.2) Swin-Tiny (timm)
    model_swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
    model_swin.to(DEVICE).eval()
    for p in model_swin.parameters(): p.requires_grad = False

    eff_feats, swin_feats, labels = [], [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            imgs = imgs.to(DEVICE)
            # EfficientNet-B0
            x = model_eff.features(imgs)
            x = model_eff.avgpool(x)
            x = torch.flatten(x, 1)
            eff_feats.append(x.cpu())
            # Swin-Tiny
            y_ = model_swin.forward_features(imgs)  # timm expose bien forward_features
            y_ = y_.mean(dim=[2,3])
            swin_feats.append(y_.cpu())
            labels.append(lbs)

    eff_feats  = torch.cat(eff_feats, 0)
    swin_feats = torch.cat(swin_feats,0)
    labels     = torch.cat(labels,   0)
    print("Shapes → Eff:", eff_feats.shape,
          "Swin:", swin_feats.shape,
          "Labels:", labels.shape)

    # =============================================================================
    # 4) DataLoader pour les MLPs
    # =============================================================================
    dl_eff  = DataLoader(FeatDataset(eff_feats,  labels),
                        batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
    dl_swin = DataLoader(FeatDataset(swin_feats, labels),
                        batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)

    # =============================================================================
    # 5) Construction & entraînement des MLPs
    # =============================================================================
    def make_mlp(input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        ).to(DEVICE)

    mlp_eff  = make_mlp(eff_feats.size(1))
    mlp_swin = make_mlp(swin_feats.size(1))
    opt_eff  = optim.Adam(mlp_eff.parameters(),  lr=LR)
    opt_swin = optim.Adam(mlp_swin.parameters(), lr=LR)
    crit     = nn.CrossEntropyLoss()

    losses = {"eff": [], "swin": []}
    for ep in range(1, EPOCHS+1):
        # EfficientNet-MLP
        mlp_eff.train()
        run = 0.
        for Xb, yb in dl_eff:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt_eff.zero_grad()
            out = mlp_eff(Xb)
            l   = crit(out, yb)
            l.backward(); opt_eff.step()
            run += l.item()
        losses["eff"].append(run / len(dl_eff))

        # Swin-MLP
        mlp_swin.train()
        run = 0.
        for Xb, yb in dl_swin:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt_swin.zero_grad()
            out = mlp_swin(Xb)
            l   = crit(out, yb)
            l.backward(); opt_swin.step()
            run += l.item()
        losses["swin"].append(run / len(dl_swin))

        print(f"[Epoch {ep}/{EPOCHS}]  "
              f"Eff-MLP loss={losses['eff'][-1]:.4f}  |  "
              f"Swin-MLP loss={losses['swin'][-1]:.4f}")

    # =============================================================================
    # 6) Sauvegardes finales
    # =============================================================================
    torch.save(eff_feats,  os.path.join(OUTPUT_DIR, "effB0_features.pt"))
    torch.save(swin_feats, os.path.join(OUTPUT_DIR, "swinT_features.pt"))
    torch.save(labels,     os.path.join(OUTPUT_DIR, "train_labels.pt"))

    torch.save(mlp_eff.state_dict(),
               os.path.join(OUTPUT_DIR, "mlp_on_effB0.pth"))
    torch.save(mlp_swin.state_dict(),
               os.path.join(OUTPUT_DIR, "mlp_on_swinT.pth"))

    pd.DataFrame({
        "epoch":          list(range(1, EPOCHS+1)),
        "loss_mlp_effB0": losses["eff"],
        "loss_mlp_swinT": losses["swin"]
    }).to_csv(os.path.join(OUTPUT_DIR, "mlp_losses.csv"), index=False)

    print("✅ Tout est sauvegardé dans", OUTPUT_DIR)


if __name__ == "__main__":
    freeze_support()
    main()
