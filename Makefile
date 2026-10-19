# Short commands for common tasks. `make help` lists them.
N ?= 5

.PHONY: help harvest harvest-status labeler mlflow test check hooks

help:
	@echo "Every pipeline command: uv run hydroknight --help"
	@echo
	@echo "make harvest [N=5]   find, judge, label and extract up to N new clips, then open a PR"
	@echo "make harvest-status  candidate queue and accepted clips"
	@echo "make labeler         restart the browser labeler (http://127.0.0.1:8765)"
	@echo "make mlflow          open the MLflow dashboard (http://127.0.0.1:5001)"
	@echo "make test            run the tests"
	@echo "make check           tests + lint + format check (what CI runs)"
	@echo "make hooks           run ruff on every commit (once per clone)"

harvest:
	bash scripts/harvest.sh $(N)

harvest-status:
	uv run hydroknight harvest status

labeler:
	-pkill -f "hydroknight labeler|hydro_knight.annotate"
	uv run hydroknight labeler

mlflow:
	uv run --extra tracking mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

test:
	uv run python -m pytest -q

check:
	uv run python -m pytest -q
	uv run ruff check .
	uv run ruff format --check .

hooks:
	uv run pre-commit install
