format:
	black .
	ruff check --fix

clean:
	rm -rf __pycache__ && rm -rf .ipynb_checkpoints && rm -rf .pytest_cache && rm -rf .ruff_cache && rm -rf battlesim.egg-info && rm -rf build