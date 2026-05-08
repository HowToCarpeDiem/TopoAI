import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import *


class BuildingDataset:
    def __init__(self, image_files, mask_files, augment=False):
        self.image_files = [Path(f) for f in image_files]
        self.mask_files = [Path(f) for f in mask_files]
        self.augment = augment
        
        assert len(self.image_files) == len(self.mask_files), \
            "Liczba obrazów i masek musi być równa"
        
        missing_images = [f for f in self.image_files if not f.exists()]
        missing_masks = [f for f in self.mask_files if not f.exists()]
        
        if missing_images:
            print(f"Brakujące obrazy: {len(missing_images)}")
            for f in missing_images[:5]:
                print(f"   - {f}")
            raise FileNotFoundError(f"Brakuje {len(missing_images)} obrazów!")
        
        if missing_masks:
            print(f"Brakujące maski: {len(missing_masks)}")
            for f in missing_masks[:5]:
                print(f"   - {f}")
            raise FileNotFoundError(f"Brakuje {len(missing_masks)} masek!")
        
        print(f"Dataset: {len(self.image_files)} par obraz-maska")
    

    def __len__(self):
        return len(self.image_files)
    

    # Wczytanie obrazów
    @staticmethod
    def load_image_numpy(image_path):
        img = Image.open(image_path).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0  
        return img
    

    # Wczytanie maski i konwersja 
    @staticmethod
    def load_mask_numpy(mask_path):
        mask = Image.open(mask_path).convert('L')  
        mask = np.array(mask, dtype=np.int32)
        
        mask_onehot = np.zeros((mask.shape[0], mask.shape[1], NUM_CLASSES), dtype=np.float32)
        for c in range(NUM_CLASSES):
            mask_onehot[:, :, c] = (mask == c).astype(np.float32)
        
        return mask_onehot
    

    @tf.function
    def augment_pair(self, image, mask):
        if not self.augment:
            return image, mask
        
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
    
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)
        
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, mask
    

    def get_tf_dataset(self, batch_size=BATCH_SIZE, shuffle=True, multi_scale=False):
        def load_pair_wrapper(img_path, mask_path):

            @tf.autograph.experimental.do_not_convert
            def load_pair_py(img_path_bytes, mask_path_bytes):
                def to_str(path_obj):
                    if isinstance(path_obj, bytes):
                        return path_obj.decode('utf-8')
                    elif isinstance(path_obj, str):
                        return path_obj
                    elif hasattr(path_obj, 'numpy'):  
                        val = path_obj.numpy()
                        if isinstance(val, bytes):
                            return val.decode('utf-8')
                        return str(val)
                    else:
                        return str(path_obj)
                
                img_path_str = to_str(img_path_bytes)
                mask_path_str = to_str(mask_path_bytes)
                
                if not Path(img_path_str).exists():
                    raise FileNotFoundError(f"Brak obrazu: {img_path_str}")
                if not Path(mask_path_str).exists():
                    raise FileNotFoundError(f"Brak maski: {mask_path_str}")
                
                img = BuildingDataset.load_image_numpy(img_path_str)
                mask = BuildingDataset.load_mask_numpy(mask_path_str)
                
                return img.astype(np.float32), mask.astype(np.float32)
            
            img, mask = tf.py_function(
                load_pair_py,
                [img_path, mask_path],
                [tf.float32, tf.float32]
            )
            
            return img, mask
        
        image_files_str = [str(f) for f in self.image_files]
        mask_files_str = [str(f) for f in self.mask_files]
        
        dataset = tf.data.Dataset.from_tensor_slices((image_files_str, mask_files_str))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_files), reshuffle_each_iteration=True)
        
        dataset = dataset.map(
            load_pair_wrapper,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.map(lambda x, y: (
            tf.ensure_shape(x, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]),
            tf.ensure_shape(y, [IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES])
        ))
        
        if self.augment:
            dataset = dataset.map(self.augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


# Wczytanie listy danych
def load_split_from_file(split_file):
    split_file = Path(split_file)
    
    if not split_file.exists():
        raise FileNotFoundError(f"Plik z podziałem nie istnieje: {split_file}")
    
    images = []
    masks = []
    
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    img_path, mask_path = parts
                    images.append(img_path)
                    masks.append(mask_path)
    
    return images, masks


def create_datasets():    
    all_images = []
    all_masks = []
    
    base_data_dir = DATA_SOURCES[0]
    raster_dirs = [d for d in base_data_dir.iterdir() if d.is_dir() and d.name != "plots" and not d.name.endswith(".txt")]
    
    for r_dir in raster_dirs:
        images_dir = r_dir / "images"
        masks_dir = r_dir / "masks_combined"
        
        if not images_dir.exists() or not masks_dir.exists():
            continue
        
        image_files = sorted(list(images_dir.glob("*.png")))
        mask_files = [masks_dir / f"{img.stem}.png" for img in image_files]
        
        valid_pairs = [
            (str(img), str(mask)) 
            for img, mask in zip(image_files, mask_files)
            if mask.exists()
        ]
        
        if valid_pairs:
            imgs, msks = zip(*valid_pairs)
            all_images.extend(imgs)
            all_masks.extend(msks)
            print(f"{r_dir.name}: {len(valid_pairs)} par")
    
    if len(all_images) == 0:
        raise ValueError("Brak danych")
    
    print(f"\nDataset: {len(all_images)} próbek")
    
    if TEST_SPLIT == 0.0:
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            all_images, all_masks, 
            test_size=VAL_SPLIT,
            random_state=42
        )
        test_imgs, test_masks = [], [] 
        
        print(f"   Train: {len(train_imgs)} ({len(train_imgs)/len(all_images)*100:.1f}%)")
        print(f"   Val:   {len(val_imgs)} ({len(val_imgs)/len(all_images)*100:.1f}%)")
    
    else:
        train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
            all_images, all_masks, 
            test_size=(VAL_SPLIT + TEST_SPLIT),
            random_state=42
        )
        
        val_imgs, test_imgs, val_masks, test_masks = train_test_split(
            temp_imgs, temp_masks,
            test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
            random_state=42
        )
        
        print(f"   Train: {len(train_imgs)} ({len(train_imgs)/len(all_images)*100:.1f}%)")
        print(f"   Val:   {len(val_imgs)} ({len(val_imgs)/len(all_images)*100:.1f}%)")
        print(f"   Test:  {len(test_imgs)} ({len(test_imgs)/len(all_images)*100:.1f}%)")
    
    splits_dir = PROJECT_ROOT / "data_splits"
    splits_dir.mkdir(exist_ok=True)
    
    for name, imgs, msks in [
        ('train', train_imgs, train_masks),
        ('val', val_imgs, val_masks),
        ('test', test_imgs, test_masks)
    ]:
        with open(splits_dir / f"{name}_files.txt", 'w') as f:
            for img, mask in zip(imgs, msks):
                f.write(f"{img}\t{mask}\n")
    
    print(f"\nPodziały zapisane: {splits_dir}")
    
    return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks


def get_datasets_from_splits(batch_size=BATCH_SIZE):
    splits_dir = PROJECT_ROOT / "data_splits"
    
    train_file = splits_dir / "train_files.txt"
    val_file = splits_dir / "val_files.txt"
    test_file = splits_dir / "test_files.txt"
    
    if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
        print("Pliki z podziałami nie istnieją. Tworzenie nowych.")
        create_datasets()
    
    print("\nŁadowanie danych z zapisanych podziałów...")
    
    train_imgs, train_masks = load_split_from_file(train_file)
    val_imgs, val_masks = load_split_from_file(val_file)
    test_imgs, test_masks = load_split_from_file(test_file)
    
    print(f"   Train: {len(train_imgs)} próbek")
    print(f"   Val:   {len(val_imgs)} próbek")
    print(f"   Test:  {len(test_imgs)} próbek")
    
    train_ds = BuildingDataset(train_imgs, train_masks, augment=True)
    val_ds = BuildingDataset(val_imgs, val_masks, augment=False)
    test_ds = BuildingDataset(test_imgs, test_masks, augment=False)
    
    train_dataset = train_ds.get_tf_dataset(batch_size=BATCH_SIZE, shuffle=True, multi_scale=False) 
    val_dataset = val_ds.get_tf_dataset(batch_size=batch_size, shuffle=False)
    test_dataset = test_ds.get_tf_dataset(batch_size=batch_size, shuffle=False)
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    print("Tworzenie podziałów")
    train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks = create_datasets()

    print("Wczytywanie z zapisanych podziałów")
    train_dataset, val_dataset, test_dataset = get_datasets_from_splits(batch_size=4)
    
    print("Testowanie batcha")
    for imgs, masks in train_dataset.take(1):
        print(f"   Batch images: {imgs.shape} (dtype: {imgs.dtype})")
        print(f"   Batch masks:  {masks.shape} (dtype: {masks.dtype})")
        print(f"   Image range:  [{imgs.numpy().min():.3f}, {imgs.numpy().max():.3f}]")
        print(f"   Mask classes: {np.unique(np.argmax(masks.numpy(), axis=-1))}")
    
    print("\nZakończono")