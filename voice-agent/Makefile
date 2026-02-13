.PHONY: install dev token-server agent web test lint format clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

token-server:
	uvicorn src.voice_agent.token_server.main:app --host 0.0.0.0 --port 8081 --reload

agent:
	python -m src.voice_agent.main

web:
	python -m http.server 5173 -d web

test:
	pytest -q

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

livekit-up:
	docker compose -f infra/livekit/docker-compose.yml up -d

livekit-down:
	docker compose -f infra/livekit/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf build dist *.egg-info
