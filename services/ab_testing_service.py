from fastapi import FastAPI, HTTPException
from router.experiments import ExperimentManager

app = FastAPI(title="A/B Testing Service")

# Initialize Experiment Manager
experiment_manager = ExperimentManager()

@app.post("/create")
def create_experiment(name: str, strategies: list, traffic_split: dict):
    try:
        experiment_manager.create_experiment(name, strategies, traffic_split)
        return {"message": f"Experiment '{name}' created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start")
def start_experiment(name: str):
    try:
        experiment_manager.start_experiment(name)
        return {"message": f"Experiment '{name}' started successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
def get_experiment_results(name: str):
    try:
        results = experiment_manager.get_experiment_results(name)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
