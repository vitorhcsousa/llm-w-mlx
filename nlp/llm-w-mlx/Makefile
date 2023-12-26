.PHONY: setup-dev
setup-dev: setup-env
	poetry run pre-commit install --hook-type pre-commit --hook-type pre-push

.PHONY: clean-env
clean-env:
	rm -rf .venv

.PHONY: setup-env
setup-env: clean-env
	poetry config virtualenvs.in-project true
	poetry env use python
	poetry install

.PHONY: format
format:
	poetry run black llm_w_mlx tests
	poetry run ruff check --fix llm_w_mlx tests

.PHONY: check
check:
	poetry run black --check llm_w_mlx tests
	poetry run bandit --exclude ".venv" -r llm_w_mlx -c "pyproject.toml"
	poetry run ruff check llm_w_mlx
	poetry run mypy llm_w_mlx


.PHONY: unit-tests
unit-tests:
	poetry run pytest tests/unit