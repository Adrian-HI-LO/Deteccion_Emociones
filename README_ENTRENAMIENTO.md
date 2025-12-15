# Guía de Entrenamiento de Modelos

Este documento explica cómo entrenar los modelos necesarios para la aplicación FaceAnalyzer-API.

## Modelos Requeridos

La aplicación requiere dos modelos entrenados:

1. **weights_keypoint.hdf5** - Detección de puntos faciales (15 puntos clave en el rostro)
2. **weights_emotions.hdf5** - Detección de emociones (5 emociones: anger, disgust, sad, happiness, surprise)

## Requisitos Previos

### 1. Descargar los Datasets

#### Dataset para Keypoints (Puntos Faciales)
- **Archivo necesario**: `data.csv`
- **Fuente**: [Kaggle - Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/data)
- **Descripción**: Contiene imágenes de rostros con 15 puntos clave anotados (30 coordenadas)
- **Colocar en**: Raíz del proyecto

#### Dataset para Emociones
- **Archivo necesario**: `icml_face_data.csv`
- **Fuente**: [Kaggle - Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- **Descripción**: Contiene imágenes de rostros con 5 categorías de emociones
- **Colocar en**: Raíz del proyecto

### 2. Estructura de Archivos

```
FaceAnalyzer-API/
├── data.csv                          # Dataset de keypoints
├── icml_face_data.csv               # Dataset de emociones
├── train_keypoints_model.py         # Script de entrenamiento keypoints
├── train_emotions_model.py          # Script de entrenamiento emociones
├── models/                          # Carpeta para guardar modelos
│   ├── weights_keypoint.hdf5       # (se genera al entrenar)
│   └── weights_emotions.hdf5       # (se genera al entrenar)
└── ...
```

## Proceso de Entrenamiento

### 1. Activar el Entorno Virtual

```bash
source Sandvenv/bin/activate
```

### 2. Entrenar Modelo de Keypoints

Este modelo detecta 15 puntos faciales clave:
- Contornos de ojos
- Cejas
- Nariz
- Boca

```bash
python train_keypoints_model.py
```

**Tiempo estimado**: 4-8 horas (depende del hardware)  
**Epochs**: 500 (con early stopping)  
**Batch size**: 32  
**Salida**: `models/weights_keypoint.hdf5`

### 3. Entrenar Modelo de Emociones

Este modelo clasifica emociones faciales:
- Anger (enojo)
- Disgust (disgusto)
- Sad (tristeza)
- Happiness (felicidad)
- Surprise (sorpresa)

```bash
python train_emotions_model.py
```

**Tiempo estimado**: 2-4 horas (depende del hardware)  
**Epochs**: 50 (con early stopping)  
**Batch size**: 64  
**Salida**: `models/weights_emotions.hdf5`

## Características de los Scripts

### train_keypoints_model.py

- **Arquitectura**: ResNet-18 modificada
- **Input**: Imágenes en escala de grises 96x96
- **Output**: 30 valores (15 puntos x 2 coordenadas)
- **Loss**: Mean Squared Error
- **Optimizer**: Adam (lr=0.0001)
- **Data augmentation**: No aplicado
- **Checkpointing**: Guarda el mejor modelo basado en val_loss

### train_emotions_model.py

- **Arquitectura**: ResNet-18 modificada
- **Input**: Imágenes en escala de grises 96x96 (redimensionadas desde 48x48)
- **Output**: 5 clases (emociones)
- **Loss**: Categorical Crossentropy
- **Optimizer**: Adam
- **Data augmentation**: 
  - Rotación (±15°)
  - Desplazamiento (10%)
  - Shear (10%)
  - Zoom (10%)
  - Volteo horizontal
- **Early stopping**: Patience de 20 epochs
- **Checkpointing**: Guarda el mejor modelo basado en val_loss

## Monitoreo del Entrenamiento

Durante el entrenamiento verás:

```
Epoch 1/500
267/267 [==============================] - 45s 168ms/step - loss: 0.0234 - accuracy: 0.7845 - val_loss: 0.0198 - val_accuracy: 0.8123
```

- **loss**: Error en el conjunto de entrenamiento
- **accuracy**: Precisión en entrenamiento
- **val_loss**: Error en el conjunto de validación
- **val_accuracy**: Precisión en validación

El modelo se guardará automáticamente cuando mejore val_loss.

## Verificación de los Modelos

Después del entrenamiento, verifica que los archivos existan:

```bash
ls -lh models/
```

Deberías ver:
- `weights_keypoint.hdf5` (~50-100 MB)
- `weights_emotions.hdf5` (~50-100 MB)

## Solución de Problemas

### Error: "No se encuentra data.csv"
- Asegúrate de descargar el dataset de Kaggle
- Coloca el archivo en la raíz del proyecto

### Error de memoria durante entrenamiento
- Reduce el batch_size en los scripts
- Cierra otras aplicaciones
- Considera usar Google Colab con GPU

### El entrenamiento es muy lento
- **Sin GPU**: Puede tomar 8-12 horas por modelo
- **Con GPU**: Reduce a 1-3 horas por modelo
- Considera usar servicios en la nube (Google Colab, Kaggle Notebooks)

### Modelos con baja precisión
- Verifica que los datasets estén completos
- Aumenta el número de epochs
- Ajusta el learning rate
- Revisa los logs de entrenamiento

## Uso de Modelos Pre-entrenados (Alternativa)

Si no deseas entrenar desde cero, puedes:

1. Buscar modelos pre-entrenados en:
   - Kaggle Datasets
   - GitHub repositorios similares
   - TensorFlow Hub

2. Asegúrate de que los modelos:
   - Tengan la misma arquitectura
   - Input shape: (96, 96, 1)
   - Outputs correctos: 30 valores (keypoints) o 5 clases (emociones)

## Continuar con la Aplicación

Una vez que tengas ambos modelos entrenados:

```bash
# Verificar que los modelos existan
ls models/

# Ejecutar la aplicación
python app.py
```

## Recursos Adicionales

- **Paper original ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Kaggle Competitions**:
  - [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection)
  - [Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

## Notas Importantes

⚠️ **El entrenamiento es intensivo en recursos**:
- Se recomienda tener al menos 8GB de RAM
- GPU acelera significativamente el proceso
- Los datasets ocupan varios GB de espacio

⚠️ **Los modelos son grandes**:
- Asegúrate de tener espacio en disco suficiente
- Los modelos no están incluidos en el repositorio (.gitignore)

⚠️ **Tiempo total estimado**:
- Descarga de datasets: 10-30 minutos
- Entrenamiento keypoints: 4-8 horas
- Entrenamiento emociones: 2-4 horas
- **Total**: 6-12 horas aproximadamente
