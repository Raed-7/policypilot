PYTHON ?= python

.PHONY: install api ui test test-fast eval pdfs up down logs

install:
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu torch
	$(PYTHON) -m pip install -r requirements-dev.txt

api:
	uvicorn policypilot.api.main:app --reload --port 8000

ui:
	cd ui && API_URL=http://localhost:8000 streamlit run app.py

test:
	$(PYTHON) -m pytest

test-fast:
	SKIP_SLOW=1 $(PYTHON) -m pytest

eval:
	$(PYTHON) -m evaluation.run

pdfs:
	$(PYTHON) scripts/build_sample_pdfs.py

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f
