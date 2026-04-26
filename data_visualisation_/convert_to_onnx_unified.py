import pickle
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort

OUTPUT_DIR = Path('output_images')
MODEL_PATH = OUTPUT_DIR / 'asfalis_lgb_v1.pkl'
SCALER_PATH = OUTPUT_DIR / 'asfalis_scaler.pkl'
ONNX_OUTPUT_PATH = OUTPUT_DIR / 'asfalis_sos_pipeline.onnx'

def run():
    print("Loading artifacts...")
    with open(MODEL_PATH, 'rb') as f:
        lgb_model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # Create Pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('lgb', lgb_model)
    ])

    # Correct initial type from onnxmltools
    initial_types = [('input', FloatTensorType([None, 17]))]

    print("Converting unified Pipeline to ONNX...")
    # This wrapper handles hybrid pipelines (SKL + LGB)
    onnx_model = onnxmltools.convert_sklearn(
        pipeline, 
        initial_types=initial_types,
        target_opset=13
    )

    # Note: onnxmltools.convert_sklearn might still use zipmap by default.
    # We can remove it manually if it exists to make mobile use easier.
    
    onnx.save(onnx_model, str(ONNX_OUTPUT_PATH))
    print(f"🎉 Success! Unified pipeline saved to {ONNX_OUTPUT_PATH}")

    # Verification
    print("Verifying pipeline...")
    sess = ort.InferenceSession(str(ONNX_OUTPUT_PATH))
    dummy_input = np.random.randn(1, 17).astype(np.float32)
    res = sess.run(None, {'input': dummy_input})
    print(f"Inference output: {res}")

if __name__ == "__main__":
    run()
