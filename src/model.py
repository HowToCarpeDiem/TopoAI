import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import *


def conv_block(inputs, filters, kernel_size=3, dropout=0.0):
    x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    
    return x


def encoder_block(inputs, filters, dropout=0.0):
    x = conv_block(inputs, filters, dropout=dropout)
    p = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return p, x


def decoder_block(inputs, skip_features, filters, dropout=0.0):
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters, dropout=dropout)
    return x


def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
               num_classes=NUM_CLASSES,
               filters=[64, 128, 256, 512],
               dropout=0.3):

    
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # ENCODER 
    p1, s1 = encoder_block(inputs, filters[0], dropout=0)
    p2, s2 = encoder_block(p1, filters[1], dropout=0)
    p3, s3 = encoder_block(p2, filters[2], dropout=dropout)
    p4, s4 = encoder_block(p3, filters[3], dropout=dropout)
    
    # BOTTLENECK
    bottleneck = conv_block(p4, filters[3] * 2, dropout=dropout)
    
    # DECODER 
    d4 = decoder_block(bottleneck, s4, filters[3], dropout=dropout)
    d3 = decoder_block(d4, s3, filters[2], dropout=dropout)
    d2 = decoder_block(d3, s2, filters[1], dropout=0)
    d1 = decoder_block(d2, s1, filters[0], dropout=0)
    
    # OUTPUT 
    outputs = layers.Conv2D(
        num_classes, 
        kernel_size=1, 
        activation='softmax',
        name='output_mask'
    )(d1)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='U-Net')
    
    return model


def weighted_categorical_crossentropy(class_weights):
    weights = tf.constant(list(class_weights.values()), dtype=tf.float32)
    
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weighted_ce = ce * weights
        return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))
    
    return loss


def focal_loss(class_weights, gamma=2.0, alpha=0.25):
    weights = tf.constant(list(class_weights.values()), dtype=tf.float32)
    
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        ce = -y_true * tf.math.log(y_pred)
        
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        focal_ce = focal_weight * alpha_weight * ce * weights
        
        return tf.reduce_mean(tf.reduce_sum(focal_ce, axis=-1))
    
    return loss


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def mean_iou(y_true, y_pred, num_classes=NUM_CLASSES):
    y_true_idx = tf.argmax(y_true, axis=-1)
    y_pred_idx = tf.argmax(y_pred, axis=-1)
    
    ious = []
    for cls in range(num_classes):
        true_mask = tf.equal(y_true_idx, cls)
        pred_mask = tf.equal(y_pred_idx, cls)
        
        intersection = tf.reduce_sum(tf.cast(true_mask & pred_mask, tf.float32))
        union = tf.reduce_sum(tf.cast(true_mask | pred_mask, tf.float32))
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou)
    
    return tf.reduce_mean(ious)


def create_model(summary=True):   
    model = build_unet(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        num_classes=NUM_CLASSES,
        filters=UNET_FILTERS,
        dropout=UNET_DROPOUT
    )
    
    if USE_FOCAL_LOSS:
        print(f"Użyto Focal Loss (gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})")
        loss_fn = focal_loss(CLASS_WEIGHTS, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    else:
        print(f"Użyto Weighted Categorical Crossentropy")
        loss_fn = weighted_categorical_crossentropy(CLASS_WEIGHTS)
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0  
        ),
        loss=loss_fn,
        metrics=['accuracy', dice_coefficient, mean_iou]
    )
    
    if summary:
        model.summary()
        
        total_params = model.count_params()
        print(f"\nParametry modelu: {total_params:,}")
        print(f"   Filters: {UNET_FILTERS}")
        print(f"   Dropout: {UNET_DROPOUT}")
        print(f"   Class weights: {CLASS_WEIGHTS}")
    
    return model


if __name__ == "__main__":
    model = create_model(summary=True)
    
    print("\nTest forward pass...")
    dummy_input = tf.random.normal((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    output = model(dummy_input, training=False)
    print(f"   Input:  {dummy_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Output sum per sample: {tf.reduce_sum(output, axis=[1,2,3])}")
    
    print("\nModel działa")