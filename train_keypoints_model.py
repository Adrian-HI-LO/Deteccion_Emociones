"""
Script para entrenar el modelo de detección de puntos faciales (keypoints)
Basado en el notebook facial_detection.ipynb

Requisitos:
- Dataset: DataSets/training.csv (Kaggle: facial-keypoints-detection)
- Descargar de: https://www.kaggle.com/c/facial-keypoints-detection/data
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, ZeroPadding2D, Conv2D, BatchNormalization, 
                                     Activation, MaxPooling2D, AveragePooling2D, 
                                     Flatten, Dense, Dropout, Add)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
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

def create_keypoints_model():
    """Crea el modelo de detección de puntos faciales"""
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
    X = AveragePooling2D((2, 2), name='Average_Pooling')(X)
    
    # Final layers
    X = Flatten()(X)
    X = Dense(4096, activation='relu')(X)
    X = Dropout(0.2)(X)
    X = Dense(2048, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(30, activation='relu')(X)  # 30 coordenadas (15 puntos x 2)
    
    model = Model(inputs=X_input, outputs=X)
    return model

def load_and_prepare_data(csv_path='DataSets/training.csv'):
    """Carga y prepara los datos del CSV de Kaggle"""
    print("Cargando datos...")
    df = pd.read_csv(csv_path)
    
    # Convertir las imágenes de string a array
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96, 96))
    
    # Eliminar filas con valores nulos
    df = df.dropna()
    
    print(f"Datos cargados: {len(df)} imágenes")
    
    # Preparar X (imágenes)
    images = []
    for img in df['Image']:
        images.append(img)
    X = np.array(images)
    X = X / 255.0  # Normalizar
    X = np.expand_dims(X, axis=-1)  # Añadir canal: (96, 96) -> (96, 96, 1)
    
    # Preparar y (coordenadas)
    y = df.drop('Image', axis=1).values
    
    return X, y

def main():
    # Verificar si existe el archivo de datos
    if not os.path.exists('DataSets/training.csv'):
        print("Error: No se encuentra el archivo 'DataSets/training.csv'")
        print("Por favor, descarga el dataset de:")
        print("https://www.kaggle.com/c/facial-keypoints-detection/data")
        return
    
    # Cargar y preparar datos
    X, y = load_and_prepare_data('DataSets/training.csv')
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    print(f"Datos de entrenamiento: {X_train.shape}")
    print(f"Datos de prueba: {X_test.shape}")
    
    # Crear el modelo
    print("\nCreando modelo...")
    model = create_keypoints_model()
    model.summary()
    
    # Compilar el modelo
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam, metrics=['accuracy'])
    
    # Callback para guardar el mejor modelo
    checkpointer = ModelCheckpoint(
        filepath="models/weights_keypoint.keras",
        verbose=1,
        save_best_only=True
    )
    
    # Entrenar el modelo
    print("\nEntrenando modelo...")
    print("Esto puede tomar varias horas dependiendo de tu hardware...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=500,
        validation_split=0.05,
        callbacks=[checkpointer]
    )
    
    # Evaluar el modelo
    print("\nEvaluando modelo...")
    result = model.evaluate(X_test, y_test)
    print(f"Accuracy en test: {result[1]}")
    
    # Guardar la arquitectura del modelo
    model_json = model.to_json()
    with open("models/keypoint_architecture.json", "w") as json_file:
        json_file.write(model_json)
    
    print("\n¡Entrenamiento completado!")
    print("Modelo guardado en: models/weights_keypoint.keras")

if __name__ == "__main__":
    main()
