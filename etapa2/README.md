# Etapa 2 — Arquitectura Profunda: BiLSTM + Word2Vec

## Descripción
Red neuronal recurrente bidireccional con embeddings Word2Vec entrenados sobre el corpus IMDb.

## Resultados
- **Accuracy:** 88.63%
- **F1 Macro:** 0.886
- **Épocas:** 6 / 20 (early stopping)
- **vs Baseline:** −0.46 pp (explicado por corpus W2V pequeño y early stopping agresivo)

## Archivos generados
- `metricas_etapa2_bilstm.csv`
- `tokenizer_bilstm.pkl`
- `bilstm_sentiment_model.h5`
- `word2vec_pca.png`
- `bilstm_training_curves.png`
- `bilstm_confusion_matrix.png`
- `comparativa_etapas.png`

## Ejecución
Requiere haber ejecutado la Etapa 1 primero (usa `metricas_etapa1_baseline.csv`).
