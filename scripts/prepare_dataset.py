import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm  

# Configuration
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_RATIO = (0.8, 0.2)  # 80% Train, 20% Test 
SEED = 42

def setup_directories(classes,processed_path):
    # If the directory already exists, delete it
    if processed_path.exists(): 
        shutil.rmtree(processed_path)
    
    for split in ["train", "test"]:
        for class_name in classes:
            (processed_path / split / class_name).mkdir(parents=True, exist_ok=True)

def split_data(raw_path=RAW_DATA_DIR, processed_path=PROCESSED_DATA_DIR):
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    print("Starting the preparation...")
    random.seed(SEED)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"The folder {raw_path} has not been found!")
        
    classes = [d.name for d in raw_path.iterdir() if d.is_dir()]
    
    setup_directories(classes,processed_path)
    for class_name in classes:
        images = [
                    f for f in (raw_path / class_name).glob("*.*")
                    if f.suffix.lower() in VALID_EXTENSIONS
                ]
        random.shuffle(images)
        
        # Generate image idx for each split
        split_idx = int(len(images) * SPLIT_RATIO[0])
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        # Copy to train
        for img in tqdm(train_imgs, desc=f"Train {class_name}", leave=False):
            shutil.copy2(img, processed_path / "train" / class_name / img.name)
            
        # Copy to test
        for img in tqdm(test_imgs, desc=f"Test  {class_name}", leave=False):
            shutil.copy2(img, processed_path / "test" / class_name / img.name)
            
    print(f"\n Processing over ! Data saved in {processed_path}")
    print(f"   - Train : {SPLIT_RATIO[0]*100}%")
    print(f"   - Test  : {SPLIT_RATIO[1]*100}%")

if __name__ == "__main__":
    split_data()