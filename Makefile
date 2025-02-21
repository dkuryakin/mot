fmt:
	ruff format src
	ruff check -s --fix --exit-zero src

lint: fmt
	mypy src
	ruff format src --check
	ruff check src

tunnel:
	ssh -L 8001:localhost:8001 -p 51022 root@mixaill76.ru

clean:
	rm videos/*__out.mp4 ||	rm markup/predict/*.yaml || true