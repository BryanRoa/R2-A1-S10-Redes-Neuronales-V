R2-A1-S10 Red Neuronal Siamesa, Embeddings y Triplet Loss
Implementación en PyTorch de una red neuronal siamesa para reconocimiento y comparación de rostros sintéticos, ejecutable en Google Colab.

Incluye el pipeline completo:

Generación de rostros sintéticos (4 identidades × 12 variantes)

Construcción de tripletas Anchor-Positive-Negative

Encoder convolucional que produce embeddings de 64 dimensiones (normalizados L2)

Entrenamiento con Triplet Loss (margen 0.8, optimizador Adam)

Evaluación con One-shot Learning (1 referencia por identidad) y Few-shot Learning (3 referencias + prototipo promedio)

Umbral pedagógico de verificación por distancia

Extensión opcional a fotografías reales autorizadas

Se aclara explícitamente por qué el experimento no demuestra Zero-shot real (las identidades evaluadas participaron en el entrenamiento del extractor).

Resultados obtenidos (semilla 42, PyTorch 2.11.0+cpu)
Métrica	Valor
Triplet Loss inicial → final	0.3981 → 0.0000
Distancia positiva media	0.1312
Distancia negativa media	1.5421
Umbral pedagógico	0.8367
One-shot accuracy	1.0 (44/44)
Few-shot accuracy	1.0 (36/36)
Autor
Bryan Felipe Roa Alvarado

Requisitos
torch

torchvision

numpy

matplotlib

Pillow
