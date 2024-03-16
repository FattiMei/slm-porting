PYTHON = python3.8


serial:
	$(PYTHON) python/csgs.py


regression:
	$(PYTHON) python/regression.py


cuda:
	$(PYTHON) python/example.py
