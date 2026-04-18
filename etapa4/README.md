# Etapa 4 — Componente Generativo: Chatbot DistilBERT + GPT-2

## Descripción
Sistema conversacional híbrido que combina:
- **Clasificador:** DistilBERT fine-tuned para detectar sentimiento
- **Generador:** GPT-2 base para producir respuestas en lenguaje natural

## Resultados
- **Distinct-2:** 0.827 (diversidad de bigramas)
- **Distinct-3:** 0.922 (diversidad de trigramas)
- Alta variación lingüística en las respuestas generadas

## Limitaciones conocidas
- GPT-2 base genera en inglés; respuestas en español pueden ser incoherentes
- El modelo generativo no fue fine-tuned para conversación específica
- Para producción se recomendaría mT5 o BLOOM para soporte multilingüe

## Ejecución
Requiere GPU. El clasificador usa pipeline de HuggingFace.
GPT-2 (~548MB) se descarga automáticamente desde HuggingFace Hub.
