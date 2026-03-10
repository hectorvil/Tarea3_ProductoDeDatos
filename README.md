# Pronóstico de demanda mensual con LightGBM

Este repositorio implementa un pipeline para pronosticar la **demanda mensual** por combinación **producto–tienda** usando el dataset de 1C Company (Kaggle: *Predict Future Sales*). El objetivo es apoyar decisiones de inventario: **cuánto pedir y dónde colocarlo con anticipación**, buscando reducir sobrestock y disminuir ventas perdidas, además de automatizar un proceso que podría depender de promedios móviles y ajustes manuales.

El modelo principal es un enfoque en **dos etapas**: primero estima si habrá venta con una clasificación y después estima cuántas unidades se venderán con una regresión. Esto es especialmente útil cuando la demanda es **esporádica e intermitente**.

## Resultados
- El desempeño agregado cumple la meta operativa: **RMSE cercano a 1**, lo cual está por debajo del umbral de 5 unidades.
- Aun así, el análisis por segmentos muestra que el modelo puede **subestimar picos de demanda**. En producción conviene complementar con criterio de negocio y/o otros criterios, tales como inventario de seguridad para productos prioritarios.


---


## Mejora del Caso de Uso

En esta iteración se aplicó una **calibración de probabilidad** sobre el modelo de dos etapas.

Originalmente la predicción final se calculaba como:

y_hat = probabilidad * unidades_predichas

Se detectó que la probabilidad del clasificador estaba ligeramente descalibrada, por lo que se aplicó un ajuste exponencial:

y_hat = (probabilidad ^ alpha) * unidades_predichas

Se realizó una búsqueda simple en validación y se fijó:

alpha = 0.90

**Impacto en métricas (validación interna)**

- RMSE antes: **0.882492**
- RMSE después: **0.880418**
- Mejora absoluta: **-0.002074**

La mejora resulta consistente, sin modificar arquitectura ni hiperparámetros principales.


---


## Estructura del repositorio

```text
.
├── artifacts
│   ├── logs
│   │   ├── inference_20260214_000519.log
│   │   ├── inference_20260301_122327.log
│   │   ├── prep_20260213_235935.log
│   │   ├── prep_20260301_122239.log
│   │   ├── train_20260214_000203.log
│   │   └── train_20260301_122252.log
│   └── model.joblib
├── container
│   ├── build_and_push.sh
│   ├── Dockerfile
│   ├── predictor.py
│   ├── serve
│   ├── train
│   └── wsgi.py
├── data
│   ├── inference
│   │   └── test_features.parquet
│   ├── predictions
│   │   └── submission.csv
│   ├── prep
│   │   ├── meta.json
│   │   ├── test_features.parquet
│   │   ├── test_pairs.parquet
│   │   ├── train.parquet
│   │   └── valid.parquet
│   └── raw
│       ├── sales_train.csv
│       └── test.csv
├── docs
│   └── images
│       └── pytest.png
├── import json.py
├── import os.jl
├── import os.py
├── main.py
├── notebooks
│   ├── Entendimientodelos_datosEDA.ipynb
│   ├── FeatureEngineering.ipynb
│   ├── Modeling.ipynb
│   ├── SimulationComparation.ipynb
│   └── Tarea05_BYOC_SageMaker.ipynb
├── pyproject.toml
├── README.md
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   └── config.cpython-312.pyc
│   ├── config.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── __pycache__
│   │   ├── Dockerfile
│   │   ├── inference.py
│   │   └── test
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── __pycache__
│   │   ├── Dockerfile
│   │   ├── prep.py
│   │   └── test
│   ├── training
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── __pycache__
│   │   ├── Dockerfile
│   │   ├── test
│   │   └── train.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       ├── data_validation.py
│       ├── logging_utils.py
│       └── metrics.py
├── tmp
│   ├── model
│   │   └── model.joblib
│   └── output
├── tree.txt
└── uv.lock
```

Cada step del pipeline está organizado como módulo independiente,
incluyendo sus propias pruebas unitarias.


---


## Git Workflow

Se implementó una estrategia de branching profesional alineada con prácticas de MLOps.

### Ramas principales

- `main`: versión estable lista para producción
- `development`: rama de integración continua
- `feature/*`: ramas para cada entregable o mejora específica

### Flujo aplicado

1. Crear rama `feature/*` desde `development`
2. Implementar cambios de manera incremental
3. Realizar commits atómicos utilizando **Conventional Commits**
   - Ejemplo: `feat(modelo): calibración de probabilidad`
   - Ejemplo: `test(training): agrega pruebas para RMSE`
4. Abrir Pull Request hacia `development`
5. Revisar y aprobar cambios
6. Merge a `development`
7. Pull Request final de `development` hacia `main`

### Política aplicada

- No se realizaron commits directos a `main`
- No se realizaron commits directos a `development`
- Todo cambio pasó por una feature branch y un Pull Request


---


## Calidad de Código

Para verificar linting:
<img width="1662" height="1132" alt="36134EC1-1BC8-4535-A74A-D9294D8EEA4A" src="https://github.com/user-attachments/assets/bc75fab3-ecf2-433f-b457-361d43757a5c" />


---


## Detalle

### Notebooks

- **`notebooks/Entendimientodelos_datosEDA.ipynb`**  
  Exploración del dataset: nulos, rangos, outliers, devoluciones, agregación mensual, estacionalidad e intermitencia (recency, meses con venta).

- **`notebooks/FeatureEngineering.ipynb`**  
  Construcción de features para series de tiempo (lags, ventanas recientes, recency, intermitencia, señales de precio y agregados laggeados) y guardado de base intermedia en `data/prep/`.

- **`notebooks/Modeling.ipynb`**  
  Entrenamiento del modelo final (clasificación + regresión), generación de predicciones y guardado de `artifacts/model.joblib` y `data/predictions/submission.csv`.

- **`notebooks/SimulationComparation.ipynb`**  
  Evaluación: calibración por deciles, análisis, comparación contra baseline y simulación operativa (sobrestock vs stockouts) con análisis de sensibilidad.


---


### Scripts (pipeline automatizable)

Los scripts se ejecutan desde la raíz del repo y siguen la estructura antes mencionada:

- **`src/prep.py`**  
  - Entrada: `data/raw/`  
  - Salida: `data/prep/` (train/valid/test_features + meta)

- **`src/train.py`**  
  - Entrada: `data/prep/`  
  - Salida: `artifacts/model.joblib`

- **`src/inference.py`**  
  - Entrada: `data/inference/` + `artifacts/model.joblib`  
  - Salida: `data/predictions/submission.csv`


---


## Métricas del modelo

- **Kaggle (Predict Future Sales)**
  - Public Score (RMSE): **1.01797**
  - Private Score (RMSE): **1.01588**
  - Leaderboard: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/leaderboard


---


## Dependencias principales

- **pandas**
- **numpy**
- **lightgbm**
- **scikit-learn**
- **joblib**
- **pyarrow** 
- **pytest**


---
  

## Instalación y Setup

### Clonar el repositorio:
```bash
git clone <repo_url>
cd Tarea3_ProductoDeDatos
```
### Instalar dependencias con uv:
```bash
uv sync
```
### O manualmente:
```bash
pip install pandas numpy lightgbm scikit-learn joblib pyarrow
pip install pytest
```


---


## Cómo ejecutar el pipeline con uv

### Preprocesamiento y features
```text
uv run python -m src.prep
```
### Entrenamiento
```text
uv run python -m src.train
```
### Inference batch
```text
uv run python -m src.inference
```
### Outputs esperados

- data/prep/train.parquet, data/prep/valid.parquet, data/prep/test_features.parquet, data/prep/test_pairs.parquet, data/prep/meta.json

- artifacts/model.joblib

- data/predictions/submission.csv
  

  ---
## Construcción de Imágenes Docker en EC2

A continuación se muestra evidencia de la construcción de imágenes Docker dentro de una instancia EC2.

### Build — preprocessing
<img width="1868" height="906" alt="E1692F50-E5F6-48C7-A654-191680406115" src="https://github.com/user-attachments/assets/c62ced5e-2e59-4453-983b-c001e1ae3532" />


### Build — training and inference
<img width="1610" height="1354" alt="88419B52-E881-4B2B-9C32-5E567042BA61" src="https://github.com/user-attachments/assets/32388fa3-4749-46f5-93aa-0e48e3807e08" />



---

## Ejecución de Contenedores con argumentos y logs en EC2

Los contenedores se ejecutan montando volúmenes para `data/` y `artifacts/`, y pasando argumentos por CLI.

### Run — preprocessing

<img width="1628" height="1126" alt="27997980-DFF3-4BE0-928C-070C04A66940" src="https://github.com/user-attachments/assets/521c251e-140e-43af-b0b7-de31e7a8a303" />

### Run — training con hiperparámetros

<img width="1186" height="962" alt="AE557DCE-04CB-413D-B82A-BDD7D1585C54" src="https://github.com/user-attachments/assets/4d0976ee-5dc8-44ec-99d6-6ed4ae723800" />

### Run — inference
<img width="1860" height="1210" alt="4726EBB8-2011-442A-8533-82307E06514B" src="https://github.com/user-attachments/assets/dda34b11-de45-48a5-b00f-9a2278ceb3df" />


---

## Pruebas Unitarias

Se implementaron **10 pruebas unitarias** distribuidas por step del pipeline:

- 4 en preprocessing
- 3 en training
- 3 en inference

Las pruebas verifican:

- Agregación correcta de datos
- Validación de esquema
- Tipo de datos
- Manejo de outliers (clipping)
- Cálculo correcto de RMSE
- Consistencia del output

### Ejecutar pruebas

Desde la raíz del proyecto:

```bash
pytest src/ -v
```
### Resultado esperado

collected 10 items
10 passed

### Evidencia de ejecución

![Pruebas unitarias en verde](docs/images/pytest.png)

Estas pruebas garantizan que los componentes críticos del pipeline
funcionan correctamente antes de cualquier despliegue o integración
continua.

---
### Sagemaker y contenedor BYOC


Se empaquetó el algoritmo en un contenedor BYOC compatible con SageMaker, se entrenó el modelo con un `Estimator` y se desplegó un endpoint de inferencia en tiempo real.

### Estructura del contenedor 
Se agregó el directorio `container/` para cumplir el contrato de SageMaker:

- **`train`**: wrapper de entrenamiento. Lee datos e hiperparámetros desde `/opt/ml/input/...` y guarda el modelo en `/opt/ml/model/`.
- **`serve`**: levanta el servidor de inferencia en el puerto 8080 usando Gunicorn, en lugar de ejecutar la app directamente con flask run
- **`predictor.py`**: define las rutas /ping y /invocations para verificación y predicciones.
- **`Dockerfile`**: construye la imagen BYOC para training/serving.
- **`build_and_push.sh`**: construye la imagen y la sube a Amazon ECR.

### Evidencia: imagen publicada en Amazon ECR
La imagen `predict-future-sales-byoc:latest` fue construida y subida correctamente a Amazon ECR.

<img width="2976" height="1010" alt="399B50C9-43D0-4150-85E9-2939D7AD72A2" src="https://github.com/user-attachments/assets/bffc2a58-999e-47ac-86f1-23208950b0b1" />

### Evidencia: endpoint real-time + predicciones
Se desplegó un endpoint de SageMaker y se verificó su estado **InService**. Luego se realizaron inferencias en tiempo real y se obtuvo una respuesta con `predictions`.

<img width="2152" height="1286" alt="33FB5142-75AB-4301-96E4-28B099D4B29A" src="https://github.com/user-attachments/assets/c822ec5c-0333-4e97-b8b1-5e260836ff80" />



Manokhin, V. (n.d.). Mastering modern time series forecasting: A comprehensive guide to statistical, machine learning, and deep learning models in Python (Early Access). Leanpub.

OpenAI. (2023). ChatGPT (Mar 14 version) [Large language model versión 5.2]. https://chat.openai.com/
..

<!-- dummy change: PR review test -->
