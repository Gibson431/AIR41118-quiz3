run:
	python -m venv .venv
	. ./.venv/bin/activate
	pip install -U -r requirements.txt
	./.venv/bin/python setup.py install
	./.venv/bin/python ./assignment3.py
	deactivate
