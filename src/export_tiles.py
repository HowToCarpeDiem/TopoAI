import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.plot import show
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon
import warnings
warnings.filterwarnings("ignore")

TILE_SIZE = 128
OVERLAP = 64
STEP = TILE_SIZE - OVERLAP
DOWNSCALE_FACTOR = 2
READ_TILE_SIZE = TILE_SIZE * DOWNSCALE_FACTOR
READ_STEP = (TILE_SIZE - OVERLAP) * DOWNSCALE_FACTOR

CLASS_KEYWORDS = {
    '16_polygon': 'mieszkalne',
    '19_polygon': 'publiczne',
    '17_polygon': 'gospodarcze',
    '21_polygon': 'przemyslowe'
}

def get_class_keyword(val):
    if not isinstance(val, str):
        return None
    for k in CLASS_KEYWORDS.keys():
        if k in val:
            return k
    return None

def get_black_mask_polygons(image_array, transform, min_area=5000):
    if image_array.shape[0] >= 3:
        grayscale = np.sum(image_array[:3], axis=0) / 3.0
        black_mask = (grayscale < 10).astype(np.uint8) * 255
    else:
        black_mask = (image_array[0] < 10).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=4)
    
    large_black_mask = np.zeros_like(black_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            large_black_mask[labels == i] = 255
            
    from rasterio.features import shapes
    black_polygons = []
    for geom, val in shapes(large_black_mask, transform=transform):
        if val == 255:
            poly = Polygon(geom['coordinates'][0])
            if poly.is_valid:
                black_polygons.append(poly)
            else:
                black_polygons.append(poly.buffer(0))
    
    if len(black_polygons) == 0:
        return Polygon()
        
    return gpd.GeoSeries(black_polygons).unary_union

def process_raster_and_vectors(raster_path, vectors_gdf, output_base):
    print(f"Przetwarzanie {raster_path.name}...")
    
    output_dir = output_base / raster_path.stem
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_KEYWORDS.values():
        (lbl_dir / cls).mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        transform = src.transform
        width, height = src.width, src.height
        crs = src.crs
        
        image_data = src.read()
        
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
        
        black_geom = get_black_mask_polygons(image_data, transform, min_area=5000)
        
        raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        local_gdf = vectors_gdf[vectors_gdf.intersects(raster_box)].copy()
        
        if not black_geom.is_empty:
            plots_dir = output_base / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            show(src, ax=ax, title=f"Usunięty czarny teren - {raster_path.name}")
            geom_df = gpd.GeoDataFrame({'geometry': [black_geom]}, crs=crs)
            geom_df.plot(ax=ax, facecolor='red', alpha=0.5, edgecolor='none')
            plt.savefig(plots_dir / f"black_mask_{raster_path.stem}.png", dpi=300)
            plt.close(fig)
            
            local_gdf['geometry'] = local_gdf.geometry.difference(black_geom)
            local_gdf = local_gdf[~local_gdf.is_empty & local_gdf.is_valid]
        
        local_gdf['matched_class'] = local_gdf['MERGE_SRC'].apply(get_class_keyword)
        local_gdf = local_gdf[local_gdf['matched_class'].notnull()]
        
        from rasterio.transform import Affine
        
        for y in range(0, height - READ_TILE_SIZE + 1, READ_STEP):
            for x in range(0, width - READ_TILE_SIZE + 1, READ_STEP):
                window = Window(x, y, READ_TILE_SIZE, READ_TILE_SIZE)
                raw_transform = src.window_transform(window)
                tile_transform = raw_transform * Affine.scale(DOWNSCALE_FACTOR, DOWNSCALE_FACTOR)
                
                tile_bounds = rasterio.windows.bounds(window, transform)
                tile_box = box(tile_bounds[0], tile_bounds[1], tile_bounds[2], tile_bounds[3])
                
                intersecting_poly = tile_box.difference(black_geom)
                if intersecting_poly.is_empty or intersecting_poly.area < (tile_box.area * 0.1): 
                    continue

                tile_data = src.read(window=window)
                tile_gdf = local_gdf[local_gdf.intersects(tile_box)].copy()
                tile_gdf['geometry'] = tile_gdf.intersection(tile_box)
                tile_gdf = tile_gdf[~tile_gdf.is_empty & tile_gdf.is_valid]

                tile_name = f"{raster_path.stem}_{y}_{x}"
                
                if tile_data.dtype == np.uint16:
                     tile_data = (tile_data / 256).astype(np.uint8)

                if tile_data.shape[0] >= 3:
                    img = np.moveaxis(tile_data[:3], 0, -1)
                elif has_palette and tile_data.dtype == np.uint8:
                    img = rgb_palette[tile_data[0]]
                else:
                    img = np.stack((tile_data[0],)*3, axis=-1)
                    
                # Downscaling obrazu
                img = cv2.resize(img, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
                    
                if black_geom.intersects(tile_box):
                    mask = rasterize([(black_geom, 255)], out_shape=(TILE_SIZE, TILE_SIZE), transform=tile_transform, fill=0, all_touched=True)
                    img[mask == 255] = [255, 255, 255]
                
                img_path = img_dir / f"{tile_name}.png"
                Image.fromarray(img).save(img_path)
                
                for cls_key, cls_name in CLASS_KEYWORDS.items():
                    cls_gdf = tile_gdf[tile_gdf['matched_class'] == cls_key]
                    
                    if not cls_gdf.empty:
                        shapes_to_rasterize = [(geom, idx) for idx, geom in enumerate(cls_gdf.geometry, 1)]
                        mask_array = rasterize(
                            shapes_to_rasterize,
                            out_shape=(TILE_SIZE, TILE_SIZE),
                            transform=tile_transform,
                            fill=0,
                            dtype=np.uint16
                        )
                        mask_path = lbl_dir / cls_name / f"{tile_name}.png"
                        Image.fromarray(mask_array).save(mask_path)

def main():
    datasets = [
        {
            "vectors": Path("wektory/wektory_trening_wwa/wektory_wwa_1bufor.shp"),
            "rasters": Path("rastry/rastry_trening_wwa")
        },
        {
            "vectors": Path("wektory/wektory_trening_ns/ns_1m.shp"),
            "rasters": Path("rastry/rastry_trening_ns")
        }
    ]
    
    output_base_dir = Path("dataset_tiles")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    for ds in datasets:
        vectors_path = ds["vectors"]
        raster_dir = ds["rasters"]
        
        if not vectors_path.exists():
            print(f"Nie znaleziono pliku wektorowego: {vectors_path}")
            continue
            
        print(f"\nWczytywanie wektorów: {vectors_path}...")
        gdf = gpd.read_file(vectors_path)
        
        raster_files = list(raster_dir.glob("*.tif"))
        if not raster_files:
            print(f"Brak rastrów w: {raster_dir}")
            continue
            
        for rf in raster_files:
            with rasterio.open(rf) as s:
                crs_ras = s.crs
            
            gdf_proj = gdf.to_crs(crs_ras) if gdf.crs != crs_ras else gdf
            process_raster_and_vectors(rf, gdf_proj, output_base_dir)

if __name__ == '__main__':
    main()
