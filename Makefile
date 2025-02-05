fmt:
	ruff format src
	ruff check -s --fix --exit-zero src

lint:
	mypy src
	ruff format src --check
	ruff check src