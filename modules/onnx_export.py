# convert_to_onnx.py
# pip install --upgrade scikit-learn onnx onnxruntime skl2onnx

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin, RegressorMixin
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy.testing as npt


def _find_model_file(d: Path) -> Path:
    for name in ("model.pkl", "model.pk","model.model"):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No model.pkl or model.pk in {d}")


def onnx_convert(
    algo_dir: Path,
    *,
    target_opset: int = 15,
    n_samples: int = 8,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    seed: int = 0,
    allow_mismatch: bool = False,   # True => print warn instead of raise on verify fail
) -> Path:
    """
    Convert sklearn (scaler + model) to ONNX and verify outputs.

    Expects in `algo_dir/`:
      - scaler.pkl
      - model.pkl (or model.pk)

    Returns:
      Path to saved ONNX file.
    """
    scaler_path = algo_dir / "scaler.pkl"
    model_path  = _find_model_file(algo_dir)

    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler.pkl in {algo_dir}")

    # Trusted artifacts: suppress Bandit for this converter-only script
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)  # nosec B301,B403

    with open(model_path, "rb") as f:
        model = pickle.load(f)   # nosec B301,B403

    # Build pipeline (scaler -> model)
    pipe = Pipeline([("scaler", scaler), ("model", model)])

    # Infer n_features
    if hasattr(model, "n_features_in_"):
        n_features = int(model.n_features_in_)
    elif hasattr(scaler, "n_features_in_"):
        n_features = int(scaler.n_features_in_)
    else:
        # optional fallback if you ship a JSON list of feature names/values
        input_json = algo_dir / "input.txt"
        if input_json.exists():
            with open(input_json, "r") as fh:
                features = json.load(fh)
            n_features = int(len(features))
        else:
            raise RuntimeError(
                f"Cannot infer n_features for {algo_dir}. "
                "Set n_features manually or provide input.txt (JSON list)."
            )

    # ONNX input signature
    initial_types = [("input", FloatTensorType([None, n_features]))]

    # Disable ZipMap for classifiers -> clean ndarray probs
    options: Optional[dict] = None
    if isinstance(model, ClassifierMixin):
        options = {id(pipe.named_steps["model"]): {"zipmap": False}}

    # Convert
    onx = convert_sklearn(
        pipe,
        initial_types=initial_types,
        target_opset=target_opset,
        options=options,
    )

    out_path = algo_dir / "model.onnx"
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())

    # Save a tiny metadata file (handy for debugging)
    meta = {
        "opset": target_opset,
        "n_features": n_features,
        "estimator": type(model).__name__,
        "is_classifier": isinstance(model, ClassifierMixin),
        "is_regressor": isinstance(model, RegressorMixin),
    }
    (algo_dir / "model.onnx.meta.json").write_text(json.dumps(meta, indent=2))

    # ===== Verification =====
    rng = np.random.default_rng(seed)
    X_SAMPLE = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    is_clf = isinstance(model, ClassifierMixin)
    is_reg = isinstance(model, RegressorMixin)

    try:
        # sklearn outputs
        if is_clf:
            skl_labels = pipe.predict(X_SAMPLE)
            skl_proba = pipe.predict_proba(X_SAMPLE) if hasattr(model, "predict_proba") else None
        elif is_reg:
            skl_pred = pipe.predict(X_SAMPLE)
        else:
            skl_generic = pipe.predict(X_SAMPLE)

        # onnxruntime outputs
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        onnx_outs = sess.run(None, {input_name: X_SAMPLE})

        # Compare
        if is_clf:
            # With zipmap=False -> [labels, probabilities] usually
            if len(onnx_outs) == 2:
                onnx_labels, onnx_proba = onnx_outs
            elif len(onnx_outs) == 1:
                onnx_proba = onnx_outs[0]
                onnx_labels = np.argmax(onnx_proba, axis=1)
            else:
                raise RuntimeError(f"Unexpected ONNX outputs for classifier: {len(onnx_outs)}")

            # labels
            npt.assert_array_equal(np.asarray(skl_labels).ravel(),
                                   np.asarray(onnx_labels).ravel())

            # probabilities (if available)
            if skl_proba is not None:
                npt.assert_allclose(skl_proba, onnx_proba, rtol=rtol, atol=atol)

        elif is_reg:
            onnx_pred = np.asarray(onnx_outs[0]).ravel()
            skl_pred = np.asarray(skl_pred).ravel()
            npt.assert_allclose(skl_pred, onnx_pred, rtol=rtol, atol=atol)

        else:
            onnx_primary = np.asarray(onnx_outs[0])
            skl_generic = np.asarray(skl_generic)
            if onnx_primary.shape != skl_generic.shape:
                onnx_primary = onnx_primary.reshape(skl_generic.shape)
            npt.assert_allclose(skl_generic, onnx_primary, rtol=rtol, atol=atol)

        print(f"[OK] Verified ONNX ≈ sklearn for '{algo_dir.name}' (rtol={rtol}, atol={atol})")

    except AssertionError as e:
        msg = f"[VERIFY FAIL] {algo_dir.name}: {e}"
        if allow_mismatch:
            print(msg)
        else:
            raise

    print(f"Saved ONNX to: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    ARTIFACTS_DIR = Path("./models")  # change if needed

    for sub in sorted(ARTIFACTS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        try:
            onnx_convert(sub)
        except Exception as ex:
            print(f"[SKIP] {sub.name}: {ex}")
