# Etapa 3 — Modelos Preentrenados: DistilBERT

## Descripción
Fine-tuning de DistilBERT-base-uncased sobre el dataset IMDb.
Se intentó RoBERTa-Twitter (125M params) pero fue descartado por tiempo excesivo
en Colab gratuito (2+ horas → desconexión de GPU).

## Resultados
- **Accuracy:** ~86-93% (varía según hardware Colab)
- **F1 Macro:** ~0.86-0.93
- **Épocas:** 2
- **Tiempo:** ~28-60 minutos en Colab T4

## Modelo
- `distilbert-base-uncased` (HuggingFace)
- 66M parámetros
- Fine-tuning completo, 2 épocas, batch=8, FP16

## Archivos generados
- `metricas_etapa3_distilbert.csv`
- `tabla_comparativa_etapas_1_2_3.csv`
- `distilbert_training.png`
- `distilbert_confusion.png`
- `comparativa_acumulada.png`
- `costo_vs_ganancia.png`

## Ejecución
Requiere GPU en Colab. Activar: Entorno de ejecución → Cambiar tipo → GPU T4.
