.PHONY: help setup run run-remote test demo web results clean

help:
	@echo "Model Council — make targets"
	@echo "  make setup       Interactive setup wizard, then launch web UI"
	@echo "  make web         Start FastAPI + browser on http://localhost:7860"
	@echo "  make run         Ask the council via CLI (uses .env)"
	@echo "  make run-remote  Open SSH tunnel then run"
	@echo "  make test        Run pipeline tests with mocked Ollama (no GPU)"
	@echo "  make demo        Run end-to-end demo with TINY models (.env.demo)"
	@echo "  make results     Pretty-print latest session JSON"
	@echo "  make clean       Remove results/ and reset .env"

setup:
	python3 -m setup.setup
	@$(MAKE) web

web:
	@( sleep 1.5 && python3 -c "import webbrowser; webbrowser.open('http://localhost:7860')" ) &
	uvicorn api.server:app --host 0.0.0.0 --port 7860 --reload

run:
	python3 council.py

run-remote:
	python3 -m setup.setup
	python3 council.py

test:
	python3 -m pytest tests/ -v

demo:
	python3 -m setup.setup --demo

results:
	@latest=$$(ls -t results/session_*.json 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then echo "No sessions yet."; else \
	  python3 -c "import json,sys; print(json.dumps(json.load(open('$$latest')), indent=2))"; fi

clean:
	rm -rf results/*.json .env .env.demo
	@echo "Cleaned."
