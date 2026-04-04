import os
import torch
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your custom pipeline
from src.bms_pipeline import (
    BatteryTransformer,
    run_predictor,
    run_simulator_optimiser,
    run_meta_agent,
    run_kill_agent,
)

app = FastAPI()

# 1. CRITICAL: Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FILE = os.path.join(BASE_DIR, "static", "index.html")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
GLOBALS_PATH = os.path.join(BASE_DIR, "models", "predictor_globals.pkl")

# 2. LOAD MODEL (Pre-loaded for speed)
device = torch.device("cpu")
model = BatteryTransformer(input_dim=11).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    with open(GLOBALS_PATH, "rb") as f:
        globs = pickle.load(f)
        global_mean, global_std = globs["global_mean"], globs["global_std"]
except Exception as e:
    print(f"Model error: {e}")

# 3. ROUTES
@app.get("/")
async def home():
    return FileResponse(STATIC_FILE)

@app.post("/predict")
async def predict(data: dict):
    try:
        # Neural Pipeline
        predictor_output = run_predictor(data, model, global_mean, global_std, device)
        df, transformer_state = run_simulator_optimiser(predictor_output)
        
        # Multi-Agent Logic
        selected_policy, policies, metrics_df, _ = run_meta_agent(df, transformer_state)
        final_policy, decision = run_kill_agent(df, selected_policy, transformer_state, policies, metrics_df)

        return {
            "status": "success",
            "decision": decision["decision"],
            "reason": decision["reason"],
            "confidence": float(predictor_output["confidence"]),
            "temperature": float(predictor_output["temperature"]),
            "soc": float(predictor_output["soc"]),
            "soh": float(predictor_output["soh"]),
            "policy_id": int(final_policy) if final_policy else 0
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}