python -m venv .venv
. ./.venv/bin/activate
pip install -U -r requirements.txt
pip install .

if [[ $# -eq 1 ]]; then
	./.venv/bin/python ./assignment3.py $1
else
	./.venv/bin/python ./assignment3.py
fi

deactivate
