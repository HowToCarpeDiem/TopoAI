import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import json
sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data_loader import BuildingDataset
from model import create_model


class TrainingVisualizer(keras.callbacks.Callback):
    def __init__(self, val_dataset, save_dir, frequency=5, num_samples=3):
        super().__init__()
        self.val_dataset = val_dataset
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.frequency = frequency
        self.num_samples = num_samples
        
        self.val_images = []
        self.val_masks = []
        
        print("   Zbieranie validation samples...")
        for imgs, masks in val_dataset.take(10):  
            self.val_images.append(imgs.numpy())
            self.val_masks.append(masks.numpy())
        
        self.val_images = np.concatenate(self.val_images, axis=0)
        self.val_masks = np.concatenate(self.val_masks, axis=0)
        
        building_counts = []
        for i, mask in enumerate(self.val_masks):
            mask_labels = np.argmax(mask, axis=-1)
            buildings = np.sum(mask_labels > 0)
            building_counts.append((i, buildings))
        
        building_counts.sort(key=lambda x: x[1], reverse=True)
        self.best_indices = [idx for idx, count in building_counts[:self.num_samples]]
        
        print(f"   Wybrano {self.num_samples} obrazów val")
    

    # Wizualizacja co 5 epok
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency != 0:
            return
        
        sample_images = self.val_images[self.best_indices]
        sample_masks = self.val_masks[self.best_indices]
        
        predictions = self.model.predict(sample_images, verbose=0)
        
        fig, axes = plt.subplots(self.num_samples, 3, figsize=(12, 4 * self.num_samples))
        
        if self.num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(self.num_samples):
            img = sample_images[i]
            
            # Ground truth
            mask_true = np.argmax(sample_masks[i], axis=-1)
            mask_true_colored = np.zeros((*mask_true.shape, 3), dtype=np.uint8)
            for cls, color in CLASS_COLORS.items():
                mask_true_colored[mask_true == cls] = color
            
            # Predykcja
            mask_pred = np.argmax(predictions[i], axis=-1)
            mask_pred_colored = np.zeros((*mask_pred.shape, 3), dtype=np.uint8)
            for cls, color in CLASS_COLORS.items():
                mask_pred_colored[mask_pred == cls] = color
            
            # Obrazy
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Źródłowy obraz #{self.best_indices[i]}', fontsize=10)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_true_colored)
            axes[i, 1].set_title('Zaetykietowane dane', fontsize=10)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(mask_pred_colored)
            axes[i, 2].set_title(f'Predykcja (Epoka {epoch+1})', fontsize=10)
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / f'epoch_{epoch+1:03d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Trening')
    axes[0, 0].plot(history.history['val_loss'], label='Walidacja')
    axes[0, 0].set_xlabel('Epoka')
    axes[0, 0].set_ylabel('Strata')
    axes[0, 0].set_title('Wartość funkcji straty')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Trening')
    axes[0, 1].plot(history.history['val_accuracy'], label='Walidacja')
    axes[0, 1].set_xlabel('Epoka')
    axes[0, 1].set_ylabel('Dokładność')
    axes[0, 1].set_title('Dokładność')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dice Coefficient
    axes[1, 0].plot(history.history['dice_coefficient'], label='Trening')
    axes[1, 0].plot(history.history['val_dice_coefficient'], label='Walidacja')
    axes[1, 0].set_xlabel('Epoka')
    axes[1, 0].set_ylabel("Współczynnik Dice'a (F1)")
    axes[1, 0].set_title("DWspółczynnik Dice'a")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mean IoU
    axes[1, 1].plot(history.history['mean_iou'], label='Trening')
    axes[1, 1].plot(history.history['val_mean_iou'], label='Walidacja')
    axes[1, 1].set_xlabel('Epoka')
    axes[1, 1].set_ylabel('Średnie IoU')
    axes[1, 1].set_title('Mean Intersection over Union')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wykres historii zapisany: {save_path}")


# Zapisywanie najlepszych metryk
class MetricsSaver(keras.callbacks.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = Path(save_path)
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'val_dice_coefficient': 0.0,
            'val_mean_iou': 0.0,
            'best_epoch': 0
        }
    

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_mean_iou'] > self.best_metrics['val_mean_iou']:
            self.best_metrics = {
                'val_loss': float(logs['val_loss']),
                'val_accuracy': float(logs['val_accuracy']),
                'val_dice_coefficient': float(logs['val_dice_coefficient']),
                'val_mean_iou': float(logs['val_mean_iou']),
                'train_loss': float(logs['loss']),
                'train_accuracy': float(logs['accuracy']),
                'train_dice_coefficient': float(logs['dice_coefficient']),
                'train_mean_iou': float(logs['mean_iou']),
                'best_epoch': epoch + 1,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    

    def on_train_end(self, logs=None):
        with open(self.save_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
        
        print(f"\nNajlepsze metryki zapisane: {self.save_path}")


# Głóna funkcja treningowa
def train_model():
    print("SPRZĘT")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"   GPU: {gpu.name}")
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("     Brak GPU - trening na CPU")
    print("="*60 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = MODELS_DIR / f"unet_{timestamp}"
    session_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSTART TRENINGU - Sesja: {timestamp}\n")
    
    config_dict = {
        "timestamp": timestamp,
        "img_size": (IMG_HEIGHT, IMG_WIDTH),
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "class_weights": CLASS_WEIGHTS,
        "augmentation": USE_AUGMENTATION,
        "gpu_available": len(gpus) > 0
    }
    
    with open(session_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\nFolder sesji: {session_dir}")
    print(f"\nKonfiguracja:")
    print(f"   Rozmiar obrazu: {IMG_HEIGHT}×{IMG_WIDTH}")
    print(f"   Liczba klas: {NUM_CLASSES}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epoki: {EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Augmentacja: {USE_AUGMENTATION}")
    

    print("\nŁadowanie danych...")
    from data_loader import get_datasets_from_splits
    
    train_dataset, val_dataset, test_dataset = get_datasets_from_splits(batch_size=BATCH_SIZE)
    
    # Model
    print("\nBudowanie modelu...")
    model = create_model(summary=False)
    print(f"   Parametry: {model.count_params():,}")
    
    viz_dir = session_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    viz_dir = session_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    callbacks = [
        # Wczesne zatrzymywanie
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=str(session_dir / 'best_model.keras'),
            monitor='val_mean_iou',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / timestamp),
            histogram_freq=0,
            write_graph=False
        ),
        
        # CSV Logger
        keras.callbacks.CSVLogger(
            filename=str(session_dir / 'training_log.csv'),
            append=False
        ),
        
        # Wizualizacja
        TrainingVisualizer(
            val_dataset=val_dataset,
            save_dir=viz_dir,
            frequency=5
        ),
        
        # Zapisywanie metryk
        MetricsSaver(
            save_path=session_dir / 'best_metrics.json'
        )
    ]
    
    # Trening
    print("\nSTART TRENINGU")
    if gpus:
        print(f"   Szacowany czas: ~{EPOCHS * 0.5:.0f} minut na GPU\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTRENING ZAKOŃCZONY\n")

    
    final_model_path = session_dir / 'final_model.keras'
    model.save(final_model_path)
    print(f"\nModel zapisany: {final_model_path}")
    
    plot_training_history(history, session_dir / 'training_history.png')
    
    print(f"\nNajlepsze wyniki:")
    print(f"   Val Loss: {min(history.history['val_loss']):.4f}")
    print(f"   Val Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"   Val Dice: {max(history.history['val_dice_coefficient']):.4f}")
    print(f"   Val mIoU: {max(history.history['val_mean_iou']):.4f}")
    
    print(f"\nWszystkie pliki w: {session_dir}")
    print("\nGotowe! Możesz teraz użyć modelu do predykcji.")

    summary_path = session_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"\nMODEL: unet_{timestamp}\n")
        
        f.write("NAJLEPSZE WYNIKI:\n")
        f.write(f"   Val Loss:     {min(history.history['val_loss']):.4f}\n")
        f.write(f"   Val Accuracy: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"   Val Dice:     {max(history.history['val_dice_coefficient']):.4f}\n")
        f.write(f"   Val mIoU:     {max(history.history['val_mean_iou']):.4f}\n\n")
        
        f.write("KONFIGURACJA:\n")
        f.write(f"   Data źródła:     {[d.name for d in DATA_SOURCES]}\n")
        f.write(f"   Rozmiar obrazu:  {IMG_HEIGHT}×{IMG_WIDTH}\n")
        f.write(f"   Batch size:      {BATCH_SIZE}\n")
        f.write(f"   Epoki:           {EPOCHS}\n")
        f.write(f"   Learning rate:   {LEARNING_RATE}\n")
        f.write(f"   Focal Loss:      {USE_FOCAL_LOSS}\n")
        f.write(f"   Class weights:   {CLASS_WEIGHTS}\n\n")
        
        f.write("PLIKI:\n")
        f.write(f"   Model:           best_model.keras\n")
        f.write(f"   Metryki:         best_metrics.json\n")
        f.write(f"   Historia:        training_history.png\n")
        f.write(f"   CSV log:         training_log.csv\n")
    
    print(f"Podsumowanie zapisane: {summary_path}")
    
    return model, history, session_dir


if __name__ == "__main__":
    model, history, session_dir = train_model()