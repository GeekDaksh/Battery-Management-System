import os
import sys
import torch
import pickle
import traceback
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_FILE  = os.path.join(BASE_DIR, "static", "index.html")
MODEL_PATH   = os.path.join(BASE_DIR, "models", "best_model.pt")
GLOBALS_PATH = os.path.join(BASE_DIR, "models", "predictor_globals.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ───────────────────────────────────────────────────────
_model       = None
_global_mean = None
_global_std  = None
_load_error  = None
_device      = torch.device("cpu")


def _load_model():
    global _model, _global_mean, _global_std, _load_error
    if _model is not None:
        return True
    if _load_error is not None:
        return False
    try:
        from src.bms_pipeline import BatteryTransformer
        print(f"[BMS] Loading model from {MODEL_PATH} ...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
        if not os.path.exists(GLOBALS_PATH):
            raise FileNotFoundError(f"Globals file missing: {GLOBALS_PATH}")
        m = BatteryTransformer(input_dim=11).to(_device)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        m.eval()
        with open(GLOBALS_PATH, "rb") as f:
            globs = pickle.load(f)
        _model       = m
        _global_mean = globs["global_mean"]
        _global_std  = globs["global_std"]
        print("[BMS] ✅ Model loaded successfully.")
        return True
    except Exception:
        _load_error = traceback.format_exc()
        print(f"[BMS] ❌ Model load failed:\n{_load_error}", file=sys.stderr)
        return False


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def home():
    return FileResponse(STATIC_FILE)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "load_error": _load_error,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "globals_path_exists": os.path.exists(GLOBALS_PATH),
        "files_in_models": (
            os.listdir(os.path.join(BASE_DIR, "models"))
            if os.path.isdir(os.path.join(BASE_DIR, "models"))
            else "MISSING"
        ),
    }


@app.post("/predict")
async def predict(data: dict):
    if not _load_model():
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": f"Model not ready: {_load_error}"}
        )
    try:
        from src.bms_pipeline import (
            run_predictor,
            run_simulator_optimiser,
            run_meta_agent,
            run_kill_agent,
        )

        # ── Map all possible frontend key variants → pipeline expected keys ──
        # Frontend sends: soc, soh, temp_C, current_A
        # Pipeline expects: soc, soh, temp_C, current_A (lowercase, exact)
        pipeline_input = {
            "soc":       data.get("soc",       data.get("SOC",     0.85)),
            "soh":       data.get("soh",       data.get("SOH",     0.92)),
            "temp_C":    data.get("temp_C",    data.get("temp",    28.0)),
            "current_A": data.get("current_A", data.get("current", 4.5)),
        }

        print(f"[BMS] /predict → pipeline_input: {pipeline_input}")

        predictor_output = run_predictor(pipeline_input, _model, _global_mean, _global_std, _device)
        df, transformer_state = run_simulator_optimiser(predictor_output)
        selected_policy, policies, metrics_df, _ = run_meta_agent(df, transformer_state)
        final_policy, decision = run_kill_agent(
            df, selected_policy, transformer_state, policies, metrics_df
        )

        return {
            "status":      "success",
            "decision":    decision["decision"],
            "reason":      decision["reason"],
            "confidence":  float(predictor_output["confidence"]),
            "temperature": float(predictor_output["temperature"]),
            "soc":         float(predictor_output["soc"]),
            "soh":         float(predictor_output["soh"]),
            "policy_id":   int(final_policy) if final_policy else 0,
        }

    except Exception:
        err = traceback.format_exc()
        print(f"[BMS] /predict error:\n{err}", file=sys.stderr)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": err}
        )