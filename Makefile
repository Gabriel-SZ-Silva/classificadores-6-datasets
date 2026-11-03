# Tarefas comuns. Rode `make help` para a lista.
#
# Variáveis que podem ser sobrescritas na linha de comando:
#   VENV      diretório do ambiente virtual (padrão: .venv)
#   N_ITER    combinações de hiperparâmetros por modelo (padrão: 10)
#   DATASETS  subconjunto, separado por espaço (ex.: DATASETS="cancer_mama estresse")

VENV     ?= .venv
PYTHON   ?= python3
N_ITER   ?= 10
DATASETS ?=

ifeq ($(OS),Windows_NT)
PY := $(VENV)/Scripts/python
else
PY := $(VENV)/bin/python
endif

DS_ARG := $(if $(DATASETS),--datasets $(DATASETS),)

.DEFAULT_GOAL := help
.PHONY: help venv install baixar executar rapido relatorio painel telas notebook test lint clean

help: ## Lista os alvos disponíveis
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

venv: $(PY) ## Cria o ambiente virtual

install: $(PY) ## Instala o pacote em modo editável, com ferramentas de desenvolvimento
	$(PY) -m pip install -e ".[dev]"

baixar: ## Baixa os 6 CSVs para data/raw
	$(PY) -m classificadores baixar $(DS_ARG)

executar: ## Roda o experimento completo e gera resultados/ (10 a 15 min)
	$(PY) -m classificadores executar --n-iter $(N_ITER) $(DS_ARG)

rapido: ## Versão rápida para testar (2 combinações, sem curvas de aprendizado)
	$(PY) -m classificadores executar --n-iter 2 --sem-curvas $(DS_ARG)

relatorio: ## Regera resultados/ a partir da última execução, sem treinar de novo
	$(PY) -m classificadores relatorio

painel: ## Abre o painel HTML no navegador
	$(PY) -c "import webbrowser, pathlib; webbrowser.open(pathlib.Path('resultados/painel.html').resolve().as_uri())"

telas: ## Recaptura as telas do painel para docs/assets/telas (precisa do Playwright)
	$(PY) scripts/capturar_telas.py

notebook: ## Regenera o notebook do Colab a partir do código em src/
	$(PY) scripts/gerar_notebook.py

test: ## Roda os testes
	$(PY) -m pytest -q

lint: ## Roda o ruff
	$(PY) -m ruff check src tests scripts

clean: ## Apaga caches (não apaga resultados/)
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
