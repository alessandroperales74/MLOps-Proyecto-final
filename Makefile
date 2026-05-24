run:
	python main.py

api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8080

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

docker-build:
	docker build -t ataque-cardiaco-api .

docker-run:
	docker run -p 8080:8080 ataque-cardiaco-api

dvc:
	dvc repro