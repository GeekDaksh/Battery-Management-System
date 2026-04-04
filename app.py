import os
import pickle
import torch
import uvicorn
from fastapi import FastAPI, Request
 bondagefrom fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your custom pipeline logic
from src.bms_pipeline import (
    BatteryTransformer,
    run_predictor,
    run_simulator_optimiser,
    run_meta_agent,
    run_kill_agent,
)

app = FastAPI(title="BMS Sentinel AI")

# Enable CORS for flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
GLOBALS_PATH = os.path.join(BASE_DIR, "models", "predictor_globals.pkl")
STATIC_HTML = os.path.join(BASE_DIR, "static", "index.html")

# Model Loading (Global Scope)
device = torch.device("cpu")
model = BatteryTransformer(input_dim=11).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    with open(GLOBALS_PATH, "rb") as f:
        globs = pickle.load(f)
        global_mean = globs["global_mean"]
        global_std = globs["global_std"]
    print("✅ BMS AI Core Initialized Successfully")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

# --- ROUTES ---

@app.get("/")
async def serve_frontend():
    """Serves the Investor-Grade UI directly."""
    if not os.path.exists(STATIC_HTML):
        return {"error": f"Frontend missing at {STATIC_HTML}. Create a /static folder."}
    return FileResponse(STATIC_HTML)

@app.get("/health")
async def health():
    return {"status": "online", "engine": "Transformer-v3"}

@app.post("/predict")
async def predict(data: dict):
    try:
        # 1. Parse Telemetry
        battery_input = {
            "soc": data.get("soc", 0.5),
            "soh": data.get("soh", 0.9),
            "temp_C": data.get("temp", 25.0),
            "current_A": data.get("current", 0.0),
            "cycle_norm": data.get("cycle", 0.5),
        }

        # 2. Multi-Agent Pipeline Execution
        predictor_output = run_predictor(battery_input, model, global_mean, global_std, device)
        
        df, transformer_state = run_simulator_optimiser(predictor_output)
        transformer_state["confidence"] = predictor_output["confidence"]

        selected_policy, policies, metrics_df, policy_choices = run_meta_agent(
            df, transformer_state, mode=data.get("mode", "auto")
        )

        final_policy, decision = run_kill_agent(
            df, selected_policy, transformer_state, policies, metrics_df
        )

        # 3. JSON Payload for React Charts
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
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)