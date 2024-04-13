python -m venv .venv
. ./.venv/bin/activate
pip install -U -r requirements.txt
pip install .
./.venv/bin/python ./main.py
deactivate
