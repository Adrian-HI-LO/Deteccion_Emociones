"""
Script para entrenar el modelo de detección de emociones
Basado en el notebook facial_detection.ipynb

Requisitos:
- Dataset organizado en carpetas por emoción en DataSets/archive/
- 7 emociones: angry, disgust, fear, happy, neutral, sad, surprise
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, ZeroPadding2D, Conv2D, BatchNormalization,
                                     Activation, MaxPooling2D, AveragePooling2D,
                                     Flatten, Dense, Add, Dropout)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def res_block(X, filter, stage):
    """Bloque residual para la red neuronal"""
    X_copy = X
    f1, f2, f3 = filter
    
    # Convolutional block
    X = Conv2D(f1, (1,1), strides=(1,1), name='res_'+str(stage)+'_conv_a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPooling2D((2,2))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_conv_a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2, kernel_size=(3,3), strides=(1,1), padding='same',
               name='res_'+str(stage)+'_conv_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_conv_b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3, kernel_size=(1,1), strides=(1,1), name='res_'+str(stage)+'_conv_c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_conv_c')(X)
    
    # Short path
    X_copy = Conv2D(f3, kernel_size=(1,1), strides=(1,1), name='res_'+str(stage)+'_conv_copy',
                    kernel_initializer=glorot_uniform(seed=0))(X_copy)
    X_copy = MaxPooling2D((2,2))(X_copy)
    X_copy = BatchNormalization(axis=3, name='bn_'+str(stage)+'_conv_copy')(X_copy)
    
    # ADD
    X = Add()([X, X_copy])
    X = Activation('relu')(X)
    
    # Identity Block 1
    X_copy = X
    X = Conv2D(f1, (1,1), strides=(1,1), name='res_'+str(stage)+'_identity_1_a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_1_a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2, kernel_size=(3,3), strides=(1,1), padding='same',
               name='res_'+str(stage)+'_identity_1_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_1_b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3, kernel_size=(1,1), strides=(1,1), name='res_'+str(stage)+'_identity_1_c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_1_c')(X)
    
    X = Add()([X, X_copy])
    X = Activation('relu')(X)
    
    # Identity Block 2
    X_copy = X
    X = Conv2D(f1, (1,1), strides=(1,1), name='res_'+str(stage)+'_identity_2_a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_2_a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2, kernel_size=(3,3), strides=(1,1), padding='same',
               name='res_'+str(stage)+'_identity_2_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_2_b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3, kernel_size=(1,1), strides=(1,1), name='res_'+str(stage)+'_identity_2_c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_'+str(stage)+'_identity_2_c')(X)
    
    X = Add()([X, X_copy])
    X = Activation('relu')(X)
    
    return X

def create_emotions_model(num_classes=7):
    """Crea el modelo de detección de emociones para 7 clases"""
    input_shape = (96, 96, 1)
    X_input = Input(input_shape)
    
    # Zero-padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7,7), strides=(2,2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)
    
    # Stage 2
    X = res_block(X, filter=[64, 64, 256], stage=2)
    
    # Stage 3
    X = res_block(X, filter=[128, 128, 512], stage=3)
    
    # Average Pooling
    X = AveragePooling2D((4, 4), name='Average_Pooling')(X)
    
    # Final layer - 7 emociones
    X = Flatten()(X)
    X = Dropout(0.5)(X)
    X = Dense(num_classes, activation='softmax', name='Dense_final', kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs=X_input, outputs=X, name='Resnet18_Emotions')
    return model

def main():
    # Configurar rutas
    train_dir = 'DataSets/archive/train'
    test_dir = 'DataSets/archive/test'
    
    # Verificar si existen los directorios
    if not os.path.exists(train_dir):
        print(f"Error: No se encuentra el directorio '{train_dir}'")
        print("Por favor, verifica que el dataset esté descargado correctamente")
        return
    
    # Crear directorio para los modelos si no existe
    os.makedirs('models', exist_ok=True)
    
    # Parámetros de entrenamiento
    img_size = (96, 96)
    batch_size = 64
    epochs = 100
    
    # Generadores de datos con data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% para validación
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Cargar datos de entrenamiento
    print("\nCargando datos de entrenamiento...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )
    
    # Cargar datos de validación
    print("Cargando datos de validación...")
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )
    
    # Cargar datos de prueba
    print("Cargando datos de prueba...")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\nClases encontradas: {train_generator.class_indices}")
    print(f"Total de imágenes de entrenamiento: {train_generator.samples}")
    print(f"Total de imágenes de validación: {validation_generator.samples}")
    print(f"Total de imágenes de prueba: {test_generator.samples}")
    
    # Crear el modelo
    num_classes = len(train_generator.class_indices)
    print(f"\nCreando modelo para {num_classes} clases...")
    model = create_emotions_model(num_classes=num_classes)
    model.summary()
    
    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    earlystopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=20
    )
    
    checkpointer = ModelCheckpoint(
        filepath='models/weights_emotions.keras',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Entrenar el modelo
    print("\nEntrenando modelo...")
    print("Esto puede tomar varias horas dependiendo de tu hardware...")
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[checkpointer, earlystopping, reduce_lr],
        verbose=1
    )
    
    # Evaluar el modelo
    print("\nEvaluando modelo...")
    score = model.evaluate(test_generator)
    print(f"Test Loss: {score[0]:.4f}")
    print(f"Test Accuracy: {score[1]:.4f}")
    
    # Guardar la arquitectura del modelo
    model_json = model.to_json()
    with open("models/emotion_architecture.json", "w") as json_file:
        json_file.write(model_json)
    
    print("\n¡Entrenamiento completado!")
    print("Modelo guardado en: models/weights_emotions.keras")
    print(f"Precisión final en validación: {max(history.history['val_accuracy']):.4f}")

if __name__ == "__main__":
    main()
