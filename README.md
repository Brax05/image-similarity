# Image Similarity Model

CNN ligera que compara dos imágenes y devuelve su nivel de similitud como un porcentaje (0–100%) con precisión ±0.01%.

---

## Descripción

El modelo genera un **embedding de 128 dimensiones** por imagen usando una CNN con convoluciones separables en profundidad (inspirada en MobileNet, ~400K parámetros). La similitud se calcula como la **similitud coseno** entre los dos embeddings normalizados con L2.

Entrenado con el dataset **CIFAR-10** de HuggingFace (60 000 imágenes, 10 clases).

### Arquitectura

```
Imagen (32×32×3)
    │
    ▼
Conv2D + BatchNorm + ReLU
    │
    ▼
DepthwiseConv2D × 3 bloques (separable)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(128) + L2-normalización
    │
    ▼
Embedding (128-dim)
    │
    ▼
Similitud coseno → score 0–100%
```

---

## Requisitos

- Python 3.8+
- CUDA (opcional, para entrenamiento en GPU)

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Uso

### 1. Abrir el notebook

```bash
jupyter notebook image_similarity.ipynb
```

Ejecutar todas las celdas en orden:
1. **Carga de datos** — descarga CIFAR-10 vía HuggingFace
2. **Entrenamiento** — entrena el modelo de embeddings
3. **Comparación** — compara dos imágenes a elección

### 2. Comparar imágenes propias

Dentro del notebook, la función `compare_images` acepta rutas de archivo, arrays NumPy o imágenes PIL:

```python
score = compare_images("foto_a.jpg", "foto_b.jpg")
print(f"Similitud: {score:.2f}%")

# También acepta arrays NumPy
import numpy as np
img1 = np.array(...)   # shape (H, W, 3)
img2 = np.array(...)
score = compare_images(img1, img2)

# O imágenes PIL
from PIL import Image
score = compare_images(Image.open("a.jpg"), Image.open("b.jpg"))
```

### 3. Buscar las 5 imágenes más similares

```python
top5 = find_top5_similar("consulta.jpg", database_embeddings, database_images)
```

---

## Datos de ejemplo

La carpeta `samples/` contiene imágenes de prueba en 5 categorías (avión, auto, pájaro, gato, perro).

Para descargarlas automáticamente:

```bash
python samples/load_samples.py
```

Esto crea `samples/airplane_1.jpg`, `samples/car_1.jpg`, etc. Luego puedes usarlas en el notebook para probar comparaciones entre categorías similares y distintas.

---

## Estructura del proyecto

```
image-similarity/
├── image_similarity.ipynb   # Modelo, entrenamiento y comparación
├── requirements.txt         # Dependencias Python
├── samples/
│   ├── load_samples.py      # Descarga imágenes de ejemplo
│   ├── airplane_1.jpg
│   ├── airplane_2.jpg
│   ├── car_1.jpg
│   ├── car_2.jpg
│   ├── bird_1.jpg
│   ├── bird_2.jpg
│   ├── cat_1.jpg
│   ├── cat_2.jpg
│   ├── dog_1.jpg
│   └── dog_2.jpg
└── README.md
```

---

## Resultados esperados

| Par de imágenes | Similitud esperada |
|---|---|
| Dos aviones | 75–95% |
| Avión vs auto | 30–55% |
| Dos gatos | 70–90% |
| Gato vs perro | 45–65% |
| Objetos sin relación | < 30% |

---

## Dataset

- **Nombre:** CIFAR-10
- **Fuente:** [HuggingFace Datasets](https://huggingface.co/datasets/cifar10)
- **Tamaño:** 60 000 imágenes (50 000 entrenamiento / 10 000 test)
- **Clases:** avión, auto, pájaro, gato, ciervo, perro, rana, caballo, barco, camión
