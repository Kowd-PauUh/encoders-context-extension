venv: 
	python3 -m venv .venv
	bash -c "source .venv/bin/activate && pip install -r requirements.txt"

notebook:
	bash -c "source .venv/bin/activate && jupyter lab --port=8510 --ip=0.0.0.0 --NotebookApp.token=''"
