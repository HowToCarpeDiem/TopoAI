import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

CLASS_COLORS = {
    'mieszkalne': [255, 0, 0],      # Czerwony
    'publiczne': [0, 255, 0],       # Zielony
    'gospodarcze': [0, 0, 255],     # Niebieski
    'przemyslowe': [255, 255, 0]    # Żółty
}

def verify_full_raster(raster_dir, downscale_factor=2):
    img_dir = raster_dir / "images"
    image_files = list(img_dir.glob("*.png"))
    
    if not image_files:
        return None
        
    print(f"Odtwarzanie obrazu dla rastra: {raster_dir.name}")
    
    max_y = 0
    max_x = 0
    
    for img_path in image_files:
        parts = img_path.stem.split("_")
        x = int(parts[-1]) // downscale_factor
        y = int(parts[-2]) // downscale_factor
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        
    canvas_w = max_x + 128
    canvas_h = max_y + 128
    
    canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    mask_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    for img_path in image_files:
        parts = img_path.stem.split("_")
        x = int(parts[-1]) // downscale_factor
        y = int(parts[-2]) // downscale_factor
        tile_name = img_path.name
        
        img = np.array(Image.open(img_path))
        
        # Nakładka kolorów budynków dla danego kafelka
        tile_mask = np.zeros_like(img, dtype=np.float32)
        has_building = np.zeros((128, 128), dtype=bool)
        
        for cls, color in CLASS_COLORS.items():
            mask_path = raster_dir / "labels" / cls / tile_name
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
                binary_mask = mask > 0
                tile_mask[binary_mask] = np.array(color)
                has_building |= binary_mask
                
        # Kafelek na xy
        canvas_img[y:y+128, x:x+128] = img
        
        mask_region = canvas_mask[y:y+128, x:x+128]
        blending_region = mask_weight[y:y+128, x:x+128]
        
        # Informacje o maskach
        mask_region[has_building] = tile_mask[has_building]
        blending_region[has_building] = 1.0
        
    # Nałożenie wyników
    overlay = canvas_img.copy()
    valid_mask = mask_weight > 0
    
    overlay[valid_mask] = overlay[valid_mask] * 0.5 + canvas_mask[valid_mask] * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay

def verify_tiles():
    base_dir = Path("dataset_tiles")
    output_dir = Path("verify_tiles")
    output_dir.mkdir(exist_ok=True)
    raster_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name != "plots"]
    
    if not raster_dirs:
        print("Nie znaleziono folderów z kafelkami.")
        return
        
    sample_dirs = raster_dirs[:min(3, len(raster_dirs))]
    
    for r_dir in sample_dirs:
        overlay_img = verify_full_raster(r_dir, downscale_factor=2)
        if overlay_img is not None:
            plt.figure(figsize=(15, 15))
            plt.imshow(overlay_img)
            plt.title(f"Zrekonstruowany raster kafelków: {r_dir.name}")
            plt.axis('off')
            plt.tight_layout()
            
            # Zapis pliku poweryfikacyjnego
            out_file = output_dir / f"verify_full_{r_dir.name}.png"
            plt.savefig(out_file, dpi=300)
            print(f"Zapisano weryfikację całego obrazu jako: {out_file}")
            plt.close()

if __name__ == '__main__':
    verify_tiles()
