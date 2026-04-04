import os
import pickle
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.bms_pipeline import (
    BatteryTransformer,
    run_predictor,
    run_simulator_optimiser,
    run_meta_agent,
    run_kill_agent,
)

app = FastAPI(title="BMS AI System")

# Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
GLOBALS_PATH = os.path.join(BASE_DIR, "models", "predictor_globals.pkl")

# Model Initialization
device = torch.device("cpu")
model = BatteryTransformer(input_dim=11).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with open(GLOBALS_PATH, "rb") as f:
    globs = pickle.load(f)
    global_mean = globs["global_mean"]
    global_std = globs["global_std"]

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: dict):
    try:
        battery_input = {
            "soc": data["soc"],
            "soh": data["soh"],
            "temp_C": data["temp"],
            "current_A": data["current"],
            "cycle_norm": data.get("cycle", 0.5),
        }

        # AI Pipeline Execution
        predictor_output = run_predictor(
            battery_input,
            model,
            global_mean,
            global_std,
            device,
        )

        df, transformer_state = run_simulator_optimiser(predictor_output)
        transformer_state["confidence"] = predictor_output["confidence"]

        selected_policy, policies, metrics_df, policy_choices = run_meta_agent(
            df,
            transformer_state,
            mode=data.get("mode", "auto"),
        )

        final_policy, decision = run_kill_agent(
            df,
            selected_policy,
            transformer_state,
            policies,
            metrics_df,
        )

        return {
            "status": "success",
            "decision": decision["decision"],
            "reason": decision["reason"],
            "policy_id": int(final_policy) if final_policy is not None else None,
            "confidence": float(predictor_output["confidence"]),
            "soc": float(predictor_output["soc"]),
            "soh": float(predictor_output["soh"]),
            "temperature": float(predictor_output["temperature"]),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }