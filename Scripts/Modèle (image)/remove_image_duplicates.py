import os
import shutil
import hashlib
from collections import defaultdict
import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(__file__), '../Data/images_extracted/images') 
train_img_dir = pd.read_csv(os.path.join(DATA_PATH, "image_train"))
test_img_dir  = pd.read_csv(os.path.join(DATA_PATH, "image_test"))

# Dossiers de sortie
output_base         = os.path.join(os.path.dirname(__file__), '../Output')  
dedup_train_folder  = os.path.join(output_base, "deduped_image_train")
dedup_test_folder   = os.path.join(output_base, "deduped_image_test")
report_train_csv    = os.path.join(output_base, "dup_report_train.csv")
report_test_csv     = os.path.join(output_base, "dup_report_test.csv")

os.makedirs(dedup_train_folder, exist_ok=True)
os.makedirs(dedup_test_folder,  exist_ok=True)

# Fonction de déduplication et copie
def dedupe_and_copy(folder_in, folder_out, report_csv_path):
    hash_map = defaultdict(list)
    # a) construire map MD5 → liste de fichiers
    for fname in os.listdir(folder_in):
        if not fname.lower().endswith(".jpg"):
            continue
        src = os.path.join(folder_in, fname)
        if not os.path.isfile(src):
            continue
        with open(src, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        hash_map[h].append(fname)

    records = []
    # b) pour chaque hash, copier un exemplaire et noter les doublons
    for h, files in hash_map.items():
        files_sorted = sorted(files)
        keep    = files_sorted[0]
        removed = files_sorted[1:]
        shutil.copy(
            os.path.join(folder_in, keep),
            os.path.join(folder_out, keep)
        )
        if removed:
            records.append({
                "hash":          h,
                "keep_file":     keep,
                "removed_files": ";".join(removed)
            })

    # c) sauvegarde du rapport CSV
    pd.DataFrame(records).to_csv(report_csv_path, index=False)
    print(f"Dédoublonnage {folder_in} → {folder_out} :")
    print(f"  Total groupes de doublons : {len(records)}\n")
    return records

# Exécution pour train et test
report_train = dedupe_and_copy(train_img_dir, dedup_train_folder, report_train_csv)
report_test  = dedupe_and_copy(test_img_dir,  dedup_test_folder,  report_test_csv)

# Afficher le nombre total d’images dans chaque dossier dédupliqué
train_count = len([f for f in os.listdir(dedup_train_folder) if f.lower().endswith(".jpg")])
test_count  = len([f for f in os.listdir(dedup_test_folder)  if f.lower().endswith(".jpg")])

print(f" Nombre d'images dans '{dedup_train_folder}' : {train_count}")
print(f" Nombre d'images dans '{dedup_test_folder}'  : {test_count}")
