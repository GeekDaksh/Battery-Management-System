from fastapi import FastAPI
import torch
import pickle

from src.bms_pipeline import (
    BatteryTransformer,
    run_predictor,
    run_simulator_optimiser,
    run_meta_agent,
    run_kill_agent,
)

app = FastAPI(title="BMS AI System")

# ─────────────────────────────────────────────────────────────
# LOAD MODEL + GLOBALS (runs once at startup)
# ─────────────────────────────────────────────────────────────
device = torch.device("cpu")

MODEL_PATH = "models/best_model.pt"
GLOBALS_PATH = "models/predictor_globals.pkl"

model = BatteryTransformer(input_dim=11).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with open(GLOBALS_PATH, "rb") as f:
    globs = pickle.load(f)

global_mean = globs["global_mean"]
global_std = globs["global_std"]


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "BMS AI System Running 🚀"}


@app.post("/predict")
def predict(data: dict):
    """
    Example Input:
    {
        "soc": 0.45,
        "soh": 0.95,
        "temp": 27,
        "current": -1.5,
        "cycle": 0.5,
        "mode": "auto"
    }
    """

    battery_input = {
        "soc": data["soc"],
        "soh": data["soh"],
        "temp_C": data["temp"],
        "current_A": data["current"],
        "cycle_norm": data.get("cycle", 0.5),
    }

    # ── Agent 1: Predictor ───────────────────────────────────
    predictor_output = run_predictor(
        battery_input, model, global_mean, global_std, device
    )

    # ── Agent 2: Simulator + Optimiser ───────────────────────
    df, transformer_state = run_simulator_optimiser(predictor_output)
    transformer_state["confidence"] = predictor_output["confidence"]

    # ── Agent 3: Meta-Agent ──────────────────────────────────
    selected_policy, policies, metrics_df, policy_choices = run_meta_agent(
        df, transformer_state, mode=data.get("mode", "auto")
    )

    # ── Agent 4: Kill Agent ──────────────────────────────────
    final_policy, decision = run_kill_agent(
        df, selected_policy, transformer_state, policies, metrics_df
    )

    return {
        "decision": decision["decision"],
        "reason": decision["reason"],
        "policy_id": int(final_policy) if final_policy is not None else None,
        "confidence": predictor_output["confidence"],
        "soc": predictor_output["soc"],
        "soh": predictor_output["soh"],
        "temperature": predictor_output["temperature"],
    }