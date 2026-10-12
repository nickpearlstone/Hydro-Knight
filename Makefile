# Short commands for common tasks. `make help` lists them.
N ?= 5

.PHONY: help harvest harvest-status labeler test check

help:
	@echo "make harvest [N=5]   find, judge, label and extract up to N new clips, then open a PR"
	@echo "make harvest-status  candidate queue and accepted clips"
	@echo "make labeler         restart the browser labeler (http://127.0.0.1:8765)"
	@echo "make test            run the tests"
	@echo "make check           tests + lint + format check (what CI runs)"

harvest:
	bash scripts/harvest.sh $(N)

harvest-status:
	uv run python scripts/harvest.py status

labeler:
	-pkill -f hydro_knight.annotate
	uv run python -m hydro_knight.annotate

test:
	uv run python -m pytest -q

check:
	uv run python -m pytest -q
	uv run ruff check .
	uv run ruff format --check .
