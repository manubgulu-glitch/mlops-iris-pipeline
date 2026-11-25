# MLOps Iris Pipeline

Proyecto de la **Actividad 2.1: Estandarización de pipelines y control de calidad de datos** para el Seminario de Grado II (Especialización en Ciencia de Datos, Universidad Santo Tomás - Seccional Tunja).

Este repositorio contiene un pipeline reproducible para clasificar las especies del dataset **Iris** usando un `RandomForestClassifier`. Con un solo comando valida los datos mediante un contrato en JSON, entrena el modelo, registra los resultados en MLflow y corre las pruebas unitarias.

La validación de los datos de entrada está implementada únicamente con la biblioteca estándar de Python para verificar tipos, nulos y rangos antes de entrenar o hacer predicciones.

## Estructura del proyecto

```
mlops-iris-pipeline/
├── pipeline.py                      # Script principal que ejecuta todo el flujo
├── Makefile                         # Alternativa para ejecutar con make all
├── requirements.txt                 # Dependencias del proyecto
├── contracts/
│   └── iris_data_contract.json      # Contrato de datos (ejemplos válido e inválido)
├── src/
│   ├── config.py                    # Rutas, semilla y umbral de exactitud
│   ├── validation.py                # Lógica de validación (solo Python estándar)
│   ├── prepare.py                   # Carga, validación y división de datos (80/20)
│   ├── train.py                     # Entrenamiento y registro en MLflow
│   └── predict.py                   # Inferencia con validación previa
├── tests/
│   ├── test_validation.py           # Pruebas de validación y contrato
│   └── test_model.py                # Pruebas del modelo entrenado
├── data/                            # Datos crudos y procesados (generado al correr)
├── models/                          # Modelo serializado y métricas (generado al correr)
└── logs/                            # Logs de cada ejecución (generado al correr)
```

## Requisitos e instalación

El proyecto requiere **Python 3.12 o superior** (probado con Python 3.14.4 en Windows 11) y no necesita GPU ni servicios en la nube.

1. Clonar el repositorio y entrar a la carpeta:

```bash
git clone https://github.com/manubgulu-glitch/mlops-iris-pipeline.git
cd mlops-iris-pipeline
```

2. Crear y activar el entorno virtual:

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Con el entorno virtual activo, ejecutar el script principal desde la raíz del proyecto:

```bash
python pipeline.py
```

*(Con GNU Make instalado, el equivalente es `make all`. El `Makefile` es un envoltorio del mismo script y no se probó en Windows).*

El flujo corre 4 etapas en secuencia (`prepare` -> `validate` -> `train` -> `tests`). Si alguna etapa falla, el proceso se detiene inmediatamente, las etapas siguientes se omiten y no se entrena el modelo.

Al finalizar correctamente, la consola mostrará un resumen como este:

```text
RESUMEN
  prepare   OK          6.7s  120 train / 30 test
  validate  OK          0.0s  contrato v1.0.0, datos válidos
  train     OK         28.7s  accuracy=1.000
  tests     OK          0.2s  31 pruebas aprobadas
RESULTADO FINAL: PIPELINE COMPLETADO CON EXITO
```

### Otros comandos útiles

* Ejecutar una etapa específica: `python pipeline.py --only validate` (`prepare`, `validate`, `train` o `tests`).
* Ejecutar desde una etapa hasta el final: `python pipeline.py --from-step validate`.
* Probar otros hiperparámetros: `python pipeline.py --n-estimators 200 --max-depth 5`.
* Probar inferencia con el ejemplo válido: `python -m src.predict --example valid`.
* Verificar rechazo con el ejemplo inválido: `python -m src.predict --example invalid`.
* Correr solo las pruebas unitarias: `python -m unittest discover -s tests -t . -v` (requiere haber ejecutado antes el pipeline, porque `test_model` necesita el modelo entrenado).
* Probar solo la validación, sin entrenar: `python -m unittest tests.test_validation -v`.

### Verificación con datos inválidos

Para comprobar que el control de calidad detiene el flujo, se altera una celda de `data/processed/train.csv` (por ejemplo, se deja una celda vacía, se escribe texto en un campo numérico o se ingresa un valor fuera de rango) y se ejecuta la validación:

```bash
python pipeline.py --from-step validate
```

La etapa `validate` falla e indica la fila, el campo y la regla incumplida. Las etapas `train` y `tests` se omiten, el proceso termina con código de salida 1 y no se entrena ningún modelo:

```text
ERROR | train.csv: 3 problema(s) en 120 registro(s) (null=1, range=1, type=1)
ERROR |   [null] fila 1, campo 'sepal_width_cm': el valor es nulo o está vacío
ERROR |   [type] fila 2, campo 'sepal_length_cm': 'abc' no se puede convertir a número decimal
ERROR |   [range] fila 3, campo 'petal_length_cm': 99.9 fuera del rango permitido [1.0, 7.0]
RESUMEN
  validate  FALLO       0.0s  train.csv: 3 problema(s) en 120 registro(s) (null=1, range=1, type=1)
  train     OMITIDO     0.0s  una etapa anterior falló
  tests     OMITIDO     0.0s  una etapa anterior falló
RESULTADO FINAL: FALLO
```

Para restaurar los datos originales se ejecuta `python pipeline.py --only prepare`.

## Contrato y validación de datos

Para evitar inconsistencias entre entrenamiento y predicción (*training-serving skew*), ambas etapas usan `contracts/iris_data_contract.json`. Este archivo define los 4 campos de entrada requeridos (`sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm`), sus tipos (`float`), sus rangos permitidos en centímetros y la variable objetivo (`species` con valores 0, 1 o 2).

El módulo `src/validation.py` revisa cada registro o archivo CSV verificando:

* **Tipos de dato:** en JSON es estricto (rechaza números como texto o booleanos).
* **Nulos:** detecta celdas vacías, `null`, `NaN`, `N/A`, etc.
* **Rangos:** comprueba que las medidas estén dentro de los rangos lógicos definidos en el contrato, calculados a partir de los valores observados en Iris con un pequeño margen (ej. `petal_length_cm` entre 1.0 y 7.0 cm).
* **Esquema:** alerta si faltan columnas obligatorias o si vienen campos extra no definidos.

El contrato incluye un registro válido y uno inválido. En la etapa `validate` del pipeline se verifica automáticamente que el sistema apruebe el válido y rechace el inválido con los errores esperados.

## Seguimiento en MLflow y logs

Para revisar los experimentos registrados en MLflow:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego abrir `http://127.0.0.1:5000` en el navegador (en las versiones recientes de MLflow, pulsar primero **Model training**, arriba a la izquierda; si el puerto 5000 está ocupado, agregar `--port 5001` al comando) y entrar al experimento **Pipeline_Iris_Actividad2**, donde se guardan hiperparámetros, exactitud (`accuracy`), versión del contrato, modelo serializado y el reporte de validación.

Cada ejecución genera además una bitácora detallada en `logs/pipeline_AAAAMMDD_HHMMSS.log` (y una copia en `logs/latest.log`) con el detalle de las validaciones.

## Solución de problemas

| Síntoma | Causa probable y solución |
|---|---|
| `ModuleNotFoundError: No module named 'mlflow'` (u otra librería) | El entorno virtual no está activo o faltó `pip install -r requirements.txt`. Activar `venv` y volver a ejecutar. |
| PowerShell: *la ejecución de scripts está deshabilitada* | Ejecutar `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` y repetir la activación. |
| `No existe ...\models\model.pkl` al correr las pruebas | Aún no se ha entrenado. Ejecutar `python pipeline.py` completo. |
| Una etapa termina en `FALLO` con líneas `[null]`, `[type]` o `[range]` | Los datos incumplen el contrato; el log indica la fila y el campo. Regenerar los datos con `python pipeline.py --only prepare`. |
| `make: command not found` (Windows) | Usar `python pipeline.py`, que es equivalente. |

## Reproducibilidad y alcance

* **Reproducibilidad:** Se fijó `random_state=42` en la partición de datos (80% entrenamiento / 20% prueba) y en el modelo, con versiones bloqueadas en `requirements.txt`. Dado que Iris es un dataset pequeño y bien separado, la exactitud esperada en prueba es de `1.000`.
* **Sin dependencias externas en validación:** `validation.py` usa solo la biblioteca estándar de Python, para cumplir con los requerimientos de la actividad y facilitar la revisión del código. Las 31 pruebas usan `unittest` (también de la biblioteca estándar); las del modelo requieren además las librerías de `requirements.txt`.
* **Alcance:** El objetivo es estandarizar el flujo, automatizar pruebas y asegurar la calidad del dato, por lo que el despliegue como API queda fuera de esta entrega.

## Autoría

**Autora:** Manuela Blandón Ramírez  
Especialización en Ciencia de Datos — Universidad Santo Tomás, Seccional Tunja.
