import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import *


def create_combined_masks_for_all_sources():
    all_stats = []
    
    base_data_dir = DATA_SOURCES[0]
    raster_dirs = [d for d in base_data_dir.iterdir() if d.is_dir() and d.name != "plots" and not d.name.endswith(".txt")]
    
    for r_dir in raster_dirs:
        print(f"\nPrzetwarzanie: {r_dir.name}\n")

        images_dir = r_dir / "images"
        labels_dir = r_dir / "labels"
        masks_output_dir = r_dir / "masks_combined"
        
        if not images_dir.exists():
            print(f"Folder nie istnieje: {images_dir}")
            print("Pomijam...")
            continue
        
        masks_output_dir.mkdir(exist_ok=True)
        
        image_files = sorted([f for f in images_dir.glob("*.png")])
        print(f"Znaleziono {len(image_files)} obrazów")
        
        if len(image_files) == 0:
            print("Brak obrazów")
            continue
        
        first_img = Image.open(image_files[0])
        print(f"Przykładowy rozmiar obrazu: {first_img.size} (W×H)")
        print(f"Oczekiwany rozmiar: {IMG_WIDTH}×{IMG_HEIGHT}")
        
        if first_img.size != (IMG_WIDTH, IMG_HEIGHT):
            print(f"Niepoprawny wymiar")
            response = input("   Kontynuować dla tego źródła? (y/n): ")
            if response.lower() != 'y':
                continue
        
        stats = {
            "source": r_dir.name,
            "total": 0,
            "no_labels": 0,
            "single_class": 0,
            "multi_class": 0,
            "class_counts": {i: 0 for i in range(5)},
            "total_instances": {name: 0 for name in CLASS_FOLDERS.keys()},
            "overlap_warnings": 0
        }
        
        for idx, img_path in enumerate(tqdm(image_files, desc=f"Maski {r_dir.name}")):
            img_name = img_path.stem
            
            combined_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            
            classes_found = []
            
            # Sprawdzenie klas
            for class_name, class_id in CLASS_FOLDERS.items():
                class_folder = labels_dir / class_name
                mask_path = class_folder / f"{img_name}.png"
                
                if mask_path.exists():
                    mask_pil = Image.open(mask_path)
                    
                    if mask_pil.mode != 'L':
                        mask_pil = mask_pil.convert('L')
                    
                    instance_mask = np.array(mask_pil, dtype=np.uint16)
                    
                    if instance_mask.ndim != 2:
                        print(f"\nNieprawidłowy wymiar maski: {mask_path.name}")
                        continue
                    
                    if instance_mask.shape != (IMG_HEIGHT, IMG_WIDTH):
                        print(f"\nNiezgodny rozmiar maski: {mask_path.name}")
                        continue
                    
                    unique_instances = np.unique(instance_mask)
                    num_instances = len(unique_instances[unique_instances > 0])
                    stats["total_instances"][class_name] += num_instances
                    
                    overlap_pixels = np.sum((combined_mask > 0) & (instance_mask > 0))
                    if overlap_pixels > 0:
                        stats["overlap_warnings"] += 1
                    
                    combined_mask[instance_mask > 0] = class_id
                    classes_found.append(class_name)
            
            output_path = masks_output_dir / f"{img_name}.png"
            
            if combined_mask.shape != (IMG_HEIGHT, IMG_WIDTH):
                print(f"\nBłędny rozmiar combined_mask: {img_name}")
                continue
            
            Image.fromarray(combined_mask).save(output_path)
            
            unique_classes = np.unique(combined_mask)
            stats["total"] += 1
            
            if len(unique_classes) == 1:
                stats["no_labels"] += 1
            elif len(unique_classes) == 2:
                stats["single_class"] += 1
            else:
                stats["multi_class"] += 1
            
            for cls in unique_classes:
                stats["class_counts"][cls] += 1
        
        all_stats.append(stats)
        
        print(f"\nGOTOWE dla {r_dir.name}")
        print(f" Utworzono: {stats['total']} masek")
        print(f" Bez etykiet: {stats['no_labels']}")
        print(f" Single-class: {stats['single_class']}")
        print(f" Multi-class: {stats['multi_class']}")
        
        print(f"\nRozkład klas:")
        for cls, count in stats["class_counts"].items():
            name = CLASS_NAMES[cls]
            pct = count / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"     {cls} ({name:12s}): {count:3d} chipów ({pct:5.1f}%)")
    
    print("\nPODSUMOWANIE")

    total_chips = sum(s["total"] for s in all_stats)
    total_buildings = {name: sum(s["total_instances"][name] for s in all_stats) 
                       for name in CLASS_FOLDERS.keys()}
    
    print(f" Łącznie chipów: {total_chips}")
    for stat in all_stats:
        print(f"   - {stat['source']}: {stat['total']} chipów")
    
    print(f" Łącznie budynków:")
    total = sum(total_buildings.values())
    for class_name, count in total_buildings.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"   - {class_name:12s}: {count:4d} ({pct:5.1f}%)")
    print(f"   - {'RAZEM':12s}: {total:4d}")
    
    return all_stats


if __name__ == "__main__":
    create_combined_masks_for_all_sources()