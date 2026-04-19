# Etapa 5 — Fine-Tuning, Optimización y Despliegue

## Descripción
Etapa final del proyecto. Implementa fine-tuning eficiente y despliegue con Gradio.

## Estrategias implementadas

### Fine-Tuning Parcial
- Congela `distilbert.transformer` (97.7% de los parámetros)
- Solo entrena `pre_classifier` + `classifier` (~1.5M parámetros)
- FP16 activado para reducir uso de memoria

### LoRA (Low-Rank Adaptation)
- `r=8`, `lora_alpha=32`, aplicado a `q_lin` y `v_lin`
- Solo 0.5% de parámetros entrenables (~0.3M)
- Batch size 16 (más grande gracias al menor uso de memoria)

## Resultados
- **Fine-tuning parcial:** ~86% accuracy, ~28 min
- **LoRA:** ~86% accuracy, ~15 min

## Despliegue
Interfaz Gradio con:
- Mapeo correcto de etiquetas (LABEL_0 → NEGATIVO, LABEL_1 → POSITIVO)
- Nivel de confianza descriptivo
- 5 ejemplos precargados
- URL pública compartible (share=True)

## Bug conocido — Gradio clasifica todo como NEGATIVO
Si ocurre este problema, ejecutar antes de lanzar Gradio:

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

## Archivos generados
- `distilbert_finetuned_parcial/` — modelo y tokenizer guardados
- `tabla_comparativa_final_todas_etapas.csv`
- `partial_ft_curves.png`
- `partial_ft_confusion.png`
- `ft_comparison.png`
- `evolucion_completa.png`
