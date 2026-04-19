# Análisis de Sentimientos con Deep Learning

### Proyecto Final — Curso de Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Transformers-5.0+-yellow.svg)](https://huggingface.co)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-purple.svg)](https://gradio.app)

---

## Descripción del Proyecto

Sistema de **análisis de sentimientos con chatbot inteligente** desarrollado de forma incremental a lo largo de cinco etapas.

El sistema clasifica textos (reseñas o comentarios) como positivos o negativos y genera respuestas automáticas condicionadas al sentimiento detectado.

El enfoque del proyecto permite comparar distintos paradigmas:

- Modelos clásicos de Machine Learning
- Redes neuronales profundas
- Modelos basados en Transformers
- Técnicas modernas de fine-tuning eficiente

---

## Arquitectura del Sistema

```
Texto de entrada
      │
      ▼
┌──────────────────────────────┐
│   Clasificador de Sentimiento│
│   DistilBERT fine-tuned      │
│   (IMDb 50K reviews)         │
└────────────┬─────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
POSITIVO          NEGATIVO
    │                 │
    ▼                 ▼
┌───────────────────────────┐
│   Generador de Respuesta  │
│   GPT-2 + prompt          │
│   condicionado            │
└───────────────────────────┘
             │
             ▼
    Respuesta del chatbot
```

---

## Estructura del Repositorio

```
DeepLearningUDB/
│
│   README.md                                       ← Este archivo
│   LICENSE                                         ← Licencia MIT
│   requirements.txt                                ← Dependencias del proyecto
│
├── deploys/
│       fine-tuning.ipynb
│       vertex-api.zip
│
├── docs/
│       informe_tecnico_grupo3.pdf
│       presentacion_grupo3.pptx
│
├── etapa1/
│       Notebook1_Etapa1_Sentimientos.ipynb           ← MLP + TF-IDF (baseline)
│       README.md
│
├── etapa2/
│       Notebook2_Etapa2_BiLSTM.ipynb                 ← BiLSTM + Word2Vec
│       README.md
│
├── etapa3/
│       Notebook3_Etapa3_Transformers.ipynb           ← DistilBERT fine-tuned
│       README.md
│
├── etapa4/
│       Notebook4_Etapa4_Chatbot.ipynb                ← Chatbot DistilBERT + GPT-2
│       README.md
│
├── etapa5/
│       Notebook_Etapa5_Finetuning_Despliegue.ipynb   ← Fine-tuning parcial + LoRA + Gradio
│       README.md
│
├── media/
│   │   presentacion_proyecto_dl_grupo3.mp4
│   │
│   └── img/
│
├── models/
│   └── bilstm/
│
├── outputs/
│
└── reports/
```

---

## Resultados por Etapa

| Etapa  | Modelo                       | Accuracy   | F1 Macro | Parámetros      | Tiempo  |
| ------ | ---------------------------- | ---------- | -------- | --------------- | ------- |
| **E1** | MLP + TF-IDF                 | **89.09%** | 0.890    | ~500K           | ~30 seg |
| **E2** | BiLSTM + Word2Vec            | 88.63%     | 0.886    | ~2M             | ~30 min |
| **E3** | DistilBERT fine-tuned        | **~92.1%** | ~0.921   | 66M             | ~30 min |
| **E3** | DistilBERT fine-tuned        | **~93.9%** | ~0.939   | 125M            | ~65 min |
| **E4** | Chatbot (DistilBERT + GPT-2) | —          | —        | 66M + 117M      | —       |
| **E5** | Fine-tuning parcial          | ~86%       | ~0.860   | 66M (1.5M ent.) | ~28 min |
| **E5** | LoRA (r=8)                   | ~86%       | ~0.860   | 66M (0.3M ent.) | ~15 min |

> _Los valores de Etapa 3-5 pueden variar ligeramente según el hardware y disponibilidad de GPU en Google Colab._

---

## Dataset

| Atributo   | Detalle                                                                                     |
| ---------- | ------------------------------------------------------------------------------------------- |
| **Nombre** | IMDb Large Movie Review Dataset                                                             |
| **Fuente** | [Stanford AI Lab — Maas et al. (2011)](https://ai.stanford.edu/~amaas/data/sentiment/)      |
| **Acceso** | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Tamaño** | 50,000 reseñas en inglés                                                                    |
| **Clases** | `positive` (25,000) / `negative` (25,000)                                                   |
| **Split**  | 80% train / 20% test (`random_state=42`, estratificado)                                     |

---

## Requisitos del Sistema

- Python 3.10+
- Google Colab (recomendado) o GPU local con CUDA
- ~15 GB de espacio en disco para modelos
- RAM: mínimo 12 GB (16 GB recomendado)

---

## Demo y Recursos

### Demo interactiva:

https://huggingface.co/spaces/rugamas/udb_sentimientos

### Archivos del proyecto:

https://drive.google.com/drive/folders/18Gce0zUPXkFB7nm1biDKqRRGNhbOP8eu?usp=sharing

---

## Despliegue

El proyecto incluye una implementación de despliegue mediante interfaz web y API.

### Interfaz interactiva:

Implementada con Gradio (Etapa 5)

API basada en Next.js:

```
npx create-next-app@latest vertex-api
```

Archivos relacionados:

```
deploys/vertex-api.zip
```

---

## Instrucciones de Ejecución

### Opción 1 — Google Colab (Recomendado)

1. Abre [Google Colab](https://colab.research.google.com)
2. Ve a `Archivo → Subir notebook`
3. Sube el notebook de la etapa que quieras ejecutar
4. Activa la GPU: `Entorno de ejecución → Cambiar tipo de entorno → GPU T4`
5. Sube el dataset IMDb a `/content/IMDB-Dataset.csv` o usa:

```python
# Descargar dataset directamente en Colab
!pip install kaggle -q
# (requiere kaggle.json con tus credenciales)
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
!unzip imdb-dataset-of-50k-movie-reviews.zip
```

6. Ejecuta todas las celdas en orden (`Entorno de ejecución → Ejecutar todo`)

---

### Opción 2 — Entorno Local

#### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/sentiment-chatbot-dl.git
cd sentiment-chatbot-dl
```

#### 2. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

#### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 4. Descargar el dataset

Descarga `IMDB-Dataset.csv` desde [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) y colócalo en la raíz del proyecto o ajusta la ruta en cada notebook:

```python
# Cambiar esta línea en cada notebook según tu ruta local
df = pd.read_csv('IMDB-Dataset.csv')  # ruta local
# df = pd.read_csv('/content/IMDB-Dataset.csv')  # Google Colab
```

#### 5. Ejecutar los notebooks en orden

```bash
jupyter notebook
```

Abre y ejecuta en este orden:

1. `etapa1/Notebook1_Etapa1_Sentimientos.ipynb`
2. `etapa2/Notebook2_Etapa2_BiLSTM.ipynb`
3. `etapa3/Notebook3_Etapa3_Transformers.ipynb`
4. `etapa4/Notebook4_Etapa4_Chatbot.ipynb`
5. `etapa5/Notebook5_Etapa5_Finetuning_Despliegue.ipynb`

> **Importante:** Ejecuta los notebooks en orden ya que las etapas posteriores cargan archivos CSV de métricas generados por las anteriores.

---

## Dependencias Principales

| Librería       | Versión | Uso                        |
| -------------- | ------- | -------------------------- |
| `torch`        | 2.0+    | Backend PyTorch            |
| `transformers` | 5.0+    | DistilBERT, GPT-2, Trainer |
| `peft`         | 0.9+    | LoRA fine-tuning           |
| `gensim`       | 4.3+    | Word2Vec (Etapa 2)         |
| `tensorflow`   | 2.x     | BiLSTM con Keras (Etapa 2) |
| `gradio`       | 4.x     | Interfaz de despliegue     |
| `scikit-learn` | 1.3+    | Métricas y TF-IDF          |
| `pandas`       | 2.x     | Manejo de datos            |
| `matplotlib`   | 3.7+    | Visualizaciones            |
| `seaborn`      | 0.13+   | Heatmaps                   |

---

## Corrección Conocida — Etapa 5 (Gradio)

Si el modelo Gradio clasifica todo como NEGATIVO, ejecutar antes de lanzar la interfaz:

```python
from transformers import DistilBertForSequenceClassification

model_fix = DistilBertForSequenceClassification.from_pretrained(
    'distilbert_finetuned_parcial'
)
model_fix.config.id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
model_fix.config.label2id = {'NEGATIVE': 0, 'POSITIVE': 1}
model_fix.save_pretrained('distilbert_finetuned_parcial')
tokenizer.save_pretrained('distilbert_finetuned_parcial')
```

---

## Tecnologías Utilizadas

- **TF-IDF + MLP** — Scikit-learn
- **Word2Vec + BiLSTM** — Gensim + TensorFlow/Keras
- **DistilBERT** — HuggingFace Transformers
- **GPT-2** — HuggingFace Transformers
- **LoRA / PEFT** — HuggingFace PEFT
- **Gradio** — Interfaz interactiva de despliegue

---

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## Autores

Proyecto desarrollado como entregable final del curso de **Deep Learning**. César Bladimir Romero Rugamas, Guillermo Ulises Palacios Flores y Walter Alexander Salguero Rodríguez.

---

_Dataset IMDb: Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL 2011._
