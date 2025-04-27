import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Paramètres 
image_size = 224
batch_size = 32

# Transformation minimales pour préserver la qualité
transform_preserve = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset personnalisé
class ImageFolderFlat(Dataset):
    def __init__(self, folder_path, transform=None):
        self.filepaths = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        return self.transform(img) if self.transform else img

base_output = os.path.join(os.path.dirname(__file__), '../Output')  
train_dir   = os.path.join(base_output, "deduped_image_train")
test_dir    = os.path.join(base_output, "deduped_image_test")


train_dataset = ImageFolderFlat(train_dir, transform=transform_preserve)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImageFolderFlat(test_dir, transform=transform_preserve)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Visualisation d'une image
def show_and_save_example(loader, out_path):
    images = next(iter(loader))
    img = images[0].permute(1, 2, 0).numpy()
    # dés-normalisation
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title("Image transformée")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Exemple sauvegardé sous : {out_path}")

fig_dir = os.path.join(base_output, "figures")
os.makedirs(fig_dir, exist_ok=True)
show_and_save_example(train_loader, os.path.join(fig_dir, "example_train.png"))

# Sauvegarde des images traitées
processed_train_dir = os.path.join(base_output, "processed_images_train")
processed_test_dir  = os.path.join(base_output, "processed_images_test")
os.makedirs(processed_train_dir, exist_ok=True)
os.makedirs(processed_test_dir,  exist_ok=True)

def save_processed_images(src_folder, dst_folder, transform):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for fname in os.listdir(src_folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = Image.open(os.path.join(src_folder, fname)).convert("RGB")
        tensor = transform(img)
        arr = tensor.permute(1,2,0).numpy()
        arr = (arr * std + mean).clip(0,1)
        arr = (arr * 255).round().astype(np.uint8)
        save_path = os.path.join(dst_folder, fname)
        Image.fromarray(arr).save(save_path, format="JPEG", quality=95)

save_processed_images(train_dir, processed_train_dir, transform_preserve)
save_processed_images(test_dir,  processed_test_dir,  transform_preserve)

print("Images sauvegardées dans :")
print("   ", processed_train_dir)
print("   ", processed_test_dir)
