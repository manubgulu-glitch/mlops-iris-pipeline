# Automatización del pipeline con un solo comando:   make all
#
# Requiere GNU Make (Linux/macOS; en Windows: Git Bash, WSL o `choco install make`).
# Si no tiene make, el equivalente multiplataforma es:   python pipeline.py
#
# Use el Python del entorno virtual activo (o defina otro:  make all PYTHON=python3).
PYTHON ?= python

.PHONY: all setup prepare validate train test demo-valid demo-invalid mlflow-ui clean help

help:            ## Muestra esta ayuda
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

all:             ## Ejecuta TODO el pipeline (preparación, validación, entrenamiento y pruebas)
	$(PYTHON) pipeline.py

setup:           ## Instala las dependencias en el entorno activo
	$(PYTHON) -m pip install -r requirements.txt

prepare:         ## Solo prepara los datos (carga y división train/test)
	$(PYTHON) pipeline.py --only prepare

validate:        ## Solo valida el contrato y los datos (tipos, nulos y rangos)
	$(PYTHON) pipeline.py --only validate

train:           ## Solo entrena (requiere haber ejecutado prepare)
	$(PYTHON) pipeline.py --only train

test:            ## Solo ejecuta las pruebas (requiere haber ejecutado train)
	$(PYTHON) pipeline.py --only tests

demo-valid:      ## Predicción con la entrada válida del contrato
	$(PYTHON) -m src.predict --example valid

demo-invalid:    ## Muestra cómo se rechaza la entrada inválida del contrato
	-$(PYTHON) -m src.predict --example invalid

mlflow-ui:       ## Abre la interfaz de MLflow en http://127.0.0.1:5000
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:           ## Borra los archivos generados (datos, modelo, registros y logs)
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p,ignore_errors=True) for p in ('mlruns','__pycache__','src/__pycache__','tests/__pycache__')]; [pathlib.Path(p).unlink(missing_ok=True) for p in ('mlflow.db','models/model.pkl','models/metrics.json','data/raw/iris.csv','data/processed/train.csv','data/processed/test.csv')]; [f.unlink() for f in pathlib.Path('logs').glob('*.log')]"
