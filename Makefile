PYTHON = python3.8


serial:
	$(PYTHON) csgs.py


regression:
	$(PYTHON) regression.py


cuda:
	$(PYTHON) example.py
