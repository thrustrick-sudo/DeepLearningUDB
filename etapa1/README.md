# Etapa 1 — Modelo Base: MLP + TF-IDF

## Descripción
Establece la línea base del sistema usando vectorización TF-IDF y un clasificador MLP.

## Resultados
- **Accuracy:** 89.09%
- **F1 Macro:** 0.890
- **Iteraciones:** 13 / 200 (early stopping)
- **Errores:** 1,091 de 10,000 (FP ~540, FN ~551)

## Archivos generados
- `metricas_etapa1_baseline.csv` — métricas para etapas posteriores
- `distribucion_sentimientos.png`
- `longitud_reviews.png`
- `top_words.png`
- `curva_loss.png`
- `confusion_matrix.png`
- `metricas_modelo.png`
- `analisis_errores.png`

## Ejecución
Abrir `Notebook1_Etapa1_Sentimientos.ipynb` y ejecutar todas las celdas en orden.
Requiere `IMDB-Dataset.csv` en `/content/` (Colab) o ajustar la ruta.
