import pickle
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort

from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from lightgbm import LGBMClassifier

# Explicitly register LightGBM converter for skl2onnx with options
update_registered_converter(
    LGBMClassifier, 'LightGbmLGBMClassifier',
    calculate_linear_classifier_output_shapes, convert_lightgbm,
    options={'zipmap': [True, False], 'nocl': [True, False]}
)

OUTPUT_DIR = Path('output_images')
MODEL_PATH = OUTPUT_DIR / 'asfalis_lgb_v1.pkl'
SCALER_PATH = OUTPUT_DIR / 'asfalis_scaler.pkl'
ONNX_OUTPUT_PATH = OUTPUT_DIR / 'asfalis_sos_pipeline.onnx'

def run():
    print("Loading artifacts...")
    with open(MODEL_PATH, 'rb') as f:
        lgb_model = pickle.load(f)
    # Scaler is not needed for unified ONNX if we handle it in Android, 
    # but let's try to keep the script simple.

    print("Converting LightGBM to ONNX...")
    # onnxmltools.convert_lightgbm is the most reliable way for LGBM
    onnx_model = onnxmltools.convert_lightgbm(
        lgb_model, 
        initial_types=[('input', FloatTensorType([None, 17]))],
        target_opset=13,
        zipmap=False
    )

    onnx.save(onnx_model, str(ONNX_OUTPUT_PATH))
    print(f"🎉 Success! LightGBM model saved to {ONNX_OUTPUT_PATH}")

    # Verification
    print("Verifying pipeline...")
    sess = ort.InferenceSession(str(ONNX_OUTPUT_PATH))
    dummy_input = np.random.randn(1, 17).astype(np.float32)
    res = sess.run(None, {'input': dummy_input})
    print(f"Inference output: {res}")

if __name__ == "__main__":
    run()
