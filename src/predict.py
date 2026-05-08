import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.affinity import affine_transform
from rasterio.features import shapes
import rasterio
from rasterio.transform import from_bounds
from rasterio.transform import from_bounds, Affine 
import cv2  
from datetime import datetime 
sys.path.insert(0, str(Path(__file__).parent))
from config import *


def load_model(model_path):
    print(f"Wczytuję model: {model_path}")
    
    from model import weighted_categorical_crossentropy, dice_coefficient, mean_iou
    
    custom_objects = {
        'loss': weighted_categorical_crossentropy(CLASS_WEIGHTS),
        'dice_coefficient': dice_coefficient,
        'mean_iou': mean_iou
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model wczytany!")
    return model


def load_image(image_path, downscale_factor=1):
    image_path = Path(image_path)
    
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        print(f"Format: GeoTIFF")
        with rasterio.open(image_path) as src:
            has_palette = False
            rgb_palette = None
            try:
                cmap = src.colormap(1)
                rgb_palette = np.zeros((256, 3), dtype=np.uint8)
                for i, color in cmap.items():
                    rgb_palette[i] = [color[0], color[1], color[2]]
                has_palette = True
            except ValueError:
                pass

            if src.count >= 3:
                image = src.read([1, 2, 3]).transpose(1, 2, 0)
            else:
                image = src.read(1)
                if has_palette and image.dtype == np.uint8:
                    image = rgb_palette[image]
                else:
                    image = np.stack((image,)*3, axis=-1)
                
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            
            transform = src.transform
            crs = src.crs
            
            if downscale_factor != 1:
                new_width = image.shape[1] // downscale_factor
                new_height = image.shape[0] // downscale_factor
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                transform = transform * Affine.scale(downscale_factor, downscale_factor)
            
            print(f"   CRS: {crs}")
            print(f"   Bounds: {src.bounds}")
            
            return image, transform, crs
    
    else:
        print(f"   Format: {image_path.suffix}")
        image = np.array(Image.open(image_path).convert('RGB'))
        
        world_file_extensions = ['.pgw', '.pngw', '.jgw', '.jpgw', '.wld']
        world_file_path = None
        
        for ext in world_file_extensions:
            potential_path = image_path.with_suffix(ext)
            if potential_path.exists():
                world_file_path = potential_path
                break
        
        if world_file_path is not None:
            print(f"Georeferencing: {world_file_path.name}")
            
            try:
                with open(world_file_path, 'r') as f:
                    lines = [float(line.strip()) for line in f.readlines()]
                
                x_scale = lines[0]
                y_rotation = lines[1]
                x_rotation = lines[2]
                y_scale = lines[3]
                x_offset = lines[4]
                y_offset = lines[5]
                
                transform = Affine(x_scale, x_rotation, x_offset,
                                 y_rotation, y_scale, y_offset)
                
                print(f"   Transform: {transform}")
                
                aux_xml_path = image_path.with_suffix(image_path.suffix + '.aux.xml')
                crs = None
                
                if aux_xml_path.exists():
                    print(f"Metadata: {aux_xml_path.name}")
                    try:
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(aux_xml_path)
                        root = tree.getroot()
                        
                        srs_elem = root.find('.//SRS')
                        if srs_elem is not None and srs_elem.text:
                            from rasterio.crs import CRS
                            crs = CRS.from_string(srs_elem.text)
                            print(f"CRS: {crs}")
                        else:
                            print(f"Brak CRS - przyjmuję EPSG:2180")
                            from rasterio.crs import CRS
                            crs = CRS.from_epsg(2180)  
                    except Exception as e:
                        print(f"Błąd odczytu .aux.xml: {e}")
                        print(f"Domyślny CRS: EPSG:2180")
                        from rasterio.crs import CRS
                        crs = CRS.from_epsg(2180)
                else:
                    print(f"Brak .aux.xml - domyślny CRS: EPSG:2180")
                    from rasterio.crs import CRS
                    crs = CRS.from_epsg(2180) 
                
                if downscale_factor != 1:
                    new_width = image.shape[1] // downscale_factor
                    new_height = image.shape[0] // downscale_factor
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    transform = transform * Affine.scale(downscale_factor, downscale_factor)

                h, w = image.shape[:2]
                from rasterio.transform import array_bounds
                bounds = array_bounds(h, w, transform)
                print(f"   Bounds: {bounds}")
                print(f"   Rozmiar po downscalingu: {h}×{w} px")
                print(f"   Zasięg: {bounds[2]-bounds[0]:.1f} × {bounds[3]-bounds[1]:.1f} m")
                
                return image, transform, crs
                
            except Exception as e:
                print(f"Błąd odczytu World File: {e}")
                print(f"Współrzędne pikselowe")
                if downscale_factor != 1:
                    new_width = image.shape[1] // downscale_factor
                    new_height = image.shape[0] // downscale_factor
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                return image, None, None
        
        else:
            print(f"Brak georeferencing")
            if downscale_factor != 1:
                new_width = image.shape[1] // downscale_factor
                new_height = image.shape[0] // downscale_factor
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return image, None, None


def postprocess_mask(mask, min_size=4, open_kernel=2):
    mask_clean = np.zeros_like(mask)
    
    for class_id in range(1, NUM_CLASSES):
        binary = (mask == class_id).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_size:
                mask_clean[labels == i] = class_id
    
    return mask_clean


# Podzielenie obrazu wejściowego na kafelki 
def split_image_to_tiles(image, tile_size=128, overlap=64):
    h, w = image.shape[:2]
    stride = tile_size - overlap
    
    tiles = []
    positions = []
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))
    
    if w % stride != 0:
        for y in range(0, h - tile_size + 1, stride):
            x = w - tile_size
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))
    
    if h % stride != 0:
        for x in range(0, w - tile_size + 1, stride):
            y = h - tile_size
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))
    
    if h % stride != 0 and w % stride != 0:
        y = h - tile_size
        x = w - tile_size
        tile = image[y:y+tile_size, x:x+tile_size]
        tiles.append(tile)
        positions.append((y, x))
    
    return np.array(tiles), positions


# Połączenie kafelków
def merge_tiles_to_image(tiles, positions, output_shape, tile_size=128, overlap=64):
    h, w = output_shape[:2]
    result = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    for tile, (y, x) in zip(tiles, positions):
        result[y:y+tile_size, x:x+tile_size] += tile
        counts[y:y+tile_size, x:x+tile_size] += 1
    
    counts[counts == 0] = 1
    result = result / counts[:, :, np.newaxis]
    
    return result


def get_large_black_regions_mask(image, min_area=1000):
    grayscale = np.mean(image, axis=-1)
    black_mask = (grayscale < 10).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=4)
    
    large_black_mask = np.zeros_like(black_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            large_black_mask[labels == i] = 255
            
    return large_black_mask


def predict_large_image(model, image_path, output_dir, model_name=None, visualize=True, timestamp=None):  
    """
        model: wytrenowany model
        image_path: ścieżka do obrazu
        output_dir: folder wyjściowy
        model_name: nazwa modelu 
        visualize: czy tworzyć wizualizacje
        timestamp: timestamp sesji 
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    image_stem = Path(image_path).stem
    if model_name:
        prefix = f"{image_stem}_{model_name}_{timestamp}"
    else:
        prefix = f"{image_stem}_{timestamp}"
    
    print(f"\nPrzetwarzanie: {image_path}")
    print(f"   Timestamp: {timestamp}")
    print(f"   Prefix: {prefix}")
    
    # Wczytanie obrazu
    image, transform, crs = load_image(image_path)
    print(f"   Rozmiar: {image.shape[0]}×{image.shape[1]}")
    
    print("   Usuwanie czarnych obrzeży skanu...")
    black_mask_areas = get_large_black_regions_mask(image, min_area=1000)
    image[black_mask_areas == 255] = [255, 255, 255]
    
    tile_size = IMG_HEIGHT
    overlap = IMG_HEIGHT // 2
    
    print(f"   Tile size: {tile_size}×{tile_size}, Overlap: {overlap}")
    
    h, w = image.shape[:2]
    stride = tile_size - overlap
    positions = []
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            positions.append((y, x))
            
    if w % stride != 0:
        for y in range(0, h - tile_size + 1, stride):
            positions.append((y, w - tile_size))
            
    if h % stride != 0:
        for x in range(0, w - tile_size + 1, stride):
            positions.append((h - tile_size, x))
            
    if h % stride != 0 and w % stride != 0:
        positions.append((h - tile_size, w - tile_size))
        
    print(f"   Liczba kafelków: {len(positions)}")
    
    full_prediction = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    batch_size = 16
    for i in range(0, len(positions), batch_size):
        batch_pos = positions[i:i+batch_size]
        batch_tiles = []
        for (y, x) in batch_pos:
            batch_tiles.append(image[y:y+tile_size, x:x+tile_size])
            
        batch_tiles = np.array(batch_tiles, dtype=np.float32) / 255.0
        preds = model.predict(batch_tiles, verbose=0)
        
        for k, (y, x) in enumerate(batch_pos):
            full_prediction[y:y+tile_size, x:x+tile_size] += preds[k]
            counts[y:y+tile_size, x:x+tile_size] += 1
            
    counts[counts == 0] = 1
    full_prediction = full_prediction / counts[:, :, np.newaxis]
    
    pred_mask = np.argmax(full_prediction, axis=-1).astype(np.uint8)
    
    # Wyzerowanie przewidywań na wyczyszczonym czarnym obszarze
    pred_mask[black_mask_areas == 255] = 0

    # Post-processing
    pred_mask = postprocess_mask(pred_mask, min_size=4, open_kernel=2)

    # Zapis
    mask_path = output_dir / f"{prefix}_mask.png"
    Image.fromarray(pred_mask).save(mask_path)
    print(f"Maska zapisana: {mask_path}")
    
    if transform is not None and crs is not None:
        geotiff_path = output_dir / f"{prefix}_mask.tif"
        
        original_shape = image.shape[:2]
        if original_shape != pred_mask.shape:
            print(f"Rozmiary się różnią. Oryginalny: {original_shape}, Predykcja: {pred_mask.shape}")
            from rasterio.transform import from_bounds
            bounds = rasterio.transform.array_bounds(original_shape[0], original_shape[1], transform)
            transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, 
                                   pred_mask.shape[1], pred_mask.shape[0])
        
        with rasterio.open(
            geotiff_path,
            'w',
            driver='GTiff',
            height=pred_mask.shape[0],
            width=pred_mask.shape[1],
            count=1,
            dtype=pred_mask.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(pred_mask, 1)
        print(f"GeoTIFF zapisany: {geotiff_path}")
    
    # Wizualizacja
    if visualize:
        print("   Tworzenie wizualizacji...")
        
        mask_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for cls, color in CLASS_COLORS.items():
            mask_colored[pred_mask == cls] = color
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        axes[0].imshow(image)
        axes[0].set_title('Wejściowy obraz', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(mask_colored)
        axes[1].set_title('Wykryte budynki', fontsize=14)
        axes[1].axis('off')
        
        legend_elements = []
        for cls in range(NUM_CLASSES):
            color = np.array(CLASS_COLORS[cls]) / 255.0
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                             markerfacecolor=color, markersize=10,
                                             label=f'{cls}: {CLASS_NAMES[cls]}'))
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        axes[2].imshow(image)
        axes[2].imshow(mask_colored, alpha=0.5)
        axes[2].set_title('Nałożony wynik na wejściowy obraz', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        viz_path = output_dir / f"{prefix}_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Wizualizacja: {viz_path}")
    
    print(f"\nStatystyki predykcji:")
    for cls in range(NUM_CLASSES):
        count = np.sum(pred_mask == cls)
        pct = count / pred_mask.size * 100
        print(f"   {cls} ({CLASS_NAMES[cls]:12s}): {count:8d} px ({pct:5.1f}%)")
    
    return pred_mask, transform, crs


def mask_to_polygons(mask, transform=None, min_area=10):
    from rasterio.transform import Affine
    
    polygons = []
    
    if transform is None:
        transform = Affine.identity()
    
    for class_id in range(1, NUM_CLASSES):  
        binary_mask = (mask == class_id).astype(np.uint8)
        

        for geom, value in shapes(binary_mask, mask=binary_mask > 0, transform=transform):
            if value == 1:
                coords = geom['coordinates']
                
                if len(coords) == 1:
                    poly = Polygon(coords[0])
                else:
                    poly = Polygon(shell=coords[0], holes=coords[1:])
                
                if poly.area >= min_area:
                    polygons.append({
                        'geometry': poly,
                        'class_id': int(class_id),
                        'class_name': CLASS_NAMES[class_id],
                        'area': poly.area
                    })
    
    return polygons


def export_to_shapefile(mask, output_path, transform=None, crs='EPSG:2180', min_area=50, model_name=None, timestamp=None):
    """
        mask: numpy array (H, W)
        output_path: ścieżka .shp
        transform: affine transform z GeoTIFF (lub None dla PNG)
        crs: układ współrzędnych (lub None dla PNG bez georef)
        min_area: minimalna powierzchnia obiektu
        model_name: nazwa modelu (opcjonalnie)
        timestamp: timestamp sesji (opcjonalnie)
    """
    print(f"\nEksport do shapefile...")
    
    if transform is None:
        print(f"   Brak georeferencji (PNG) - użyto współrzędnych pikselowych")
        crs = None  
    
    polygons = mask_to_polygons(mask, transform=transform, min_area=min_area)
    print(f"   Znaleziono {len(polygons)} obiektów")
    
    
    if len(polygons) == 0:
        print("Brak obiektów do eksportu!")
        return
    
    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    
    output_path = Path(output_path)
    
    parts = [output_path.stem]
    if model_name:
        parts.append(model_name)
    if timestamp:
        parts.append(timestamp)
    
    new_stem = "_".join(parts)
    output_path = output_path.parent / f"{new_stem}{output_path.suffix}"
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    gdf.to_file(output_path)
    print(f"Shapefile zapisany: {output_path}")
    
    print(f"\nStatystyki per klasa:")
    for cls_id in range(1, NUM_CLASSES):
        cls_polys = gdf[gdf['class_id'] == cls_id]
        if len(cls_polys) > 0:
            total_area = cls_polys['area'].sum()
            
            if transform is not None and transform != Affine.identity():
                unit = "m²"
            else:
                unit = "px²"
            
            print(f"   {CLASS_NAMES[cls_id]:12s}: {len(cls_polys):4d} obiektów, {total_area:10.1f} {unit}")
    
    return gdf


def predict_on_test_set(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(PROJECT_ROOT / "data_splits" / "test_files.txt", 'r') as f:
        test_files = [line.strip().split('\t')[0] for line in f]
    
    print(f"\nPredykcja na test secie ({len(test_files)} obrazów)...")
    
    for i, img_path in enumerate(test_files[:10], 1):
        print(f"\n[{i}/10] {Path(img_path).name}")
        predict_large_image(model, img_path, output_dir / f"test_{i:03d}", visualize=True)
    
    print(f"\nWyniki w: {output_dir}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skrypt do predykcji budynków na dużych mapach rastrowych.")
    parser.add_argument("--model", type=str, required=True, help="Ścieżka do wytrenowanego modelu (.keras)")
    parser.add_argument("--input", type=str, required=True, help="Ścieżka do wejściowego pliku obrazu/rastra (.tif, .png itp.)")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "new_map"), help="Katalog wwynikowy")
    args = parser.parse_args()

    model_path = Path(args.model)
    new_map_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku modelu: {model_path}")
    if not new_map_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego: {new_map_path}")
        
    model_name = model_path.parent.name 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Model: {model_name}")
    print(f"Obraz: {new_map_path}")
    print(f"Katalog wyjściowy: {output_dir}")
    print(f"Timestamp sesji: {timestamp}")
    
    model = load_model(model_path)
    
    mask, transform, crs = predict_large_image(
        model, 
        new_map_path, 
        output_dir,
        model_name=model_name,
        visualize=True,
        timestamp=timestamp 
    )
    
    # Eksport 
    export_to_shapefile(
        mask, 
        output_dir / "buildings.shp", 
        transform=transform, 
        crs=crs, 
        min_area=0.04,
        model_name=model_name,
        timestamp=timestamp  
    )