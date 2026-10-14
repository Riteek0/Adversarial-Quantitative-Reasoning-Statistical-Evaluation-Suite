.PHONY: test catalog evaluate

test:
	pytest -v

catalog:
	python3 -m src.benchmark_catalog

evaluate:
	python3 -m src.grading_harness --generate-catalog --run-evaluation
