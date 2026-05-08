from pathlib import Path

# Ścieżki 
PROJECT_ROOT = Path(".").resolve()
DATA_SOURCES = [
    PROJECT_ROOT / "dataset_tiles"
]

MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Parametry danych
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
NUM_CLASSES = 5

CLASS_FOLDERS = {
    "mieszkalne": 1,
    "publiczne": 2,
    "gospodarcze": 3,
    "przemyslowe": 4
}

CLASS_NAMES = {
    0: "Tło",
    1: "Mieszkalny",
    2: "Publiczny",
    3: "Gospodarczy",
    4: "Przemysłowy"
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0)
}


BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-4

# Podział
VAL_SPLIT = 0.15
TEST_SPLIT = 0.00  

# Wagi klas 
CLASS_WEIGHTS = {
    0: 0.3,   
    1: 5.2,   
    2: 3.0,  
    3: 4.0,   
    4: 6.0,  
}

# Augmentacja
USE_AUGMENTATION = True

HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
ROTATION_RANGE = [0, 90, 180, 270]  

USE_BRIGHTNESS = False      
BRIGHTNESS_RANGE = 0.0      

USE_CONTRAST = False       
CONTRAST_LOWER = 1.0        
CONTRAST_UPPER = 1.0       

USE_NOISE = False           
NOISE_STDDEV = 0.0         

# Zatrzymanie
PATIENCE = 25
MIN_DELTA = 0.0005

# Focal Loss
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Architektura modelu
UNET_FILTERS = [64, 128, 256, 512]
UNET_DROPOUT = 0.25