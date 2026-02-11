import glob
import inspect
import os
import cv2
import numpy as np
import onnx
import modelopt.onnx.quantization as maq


CALIB_DIR = "calib"
CALIB_COUNT = 32
OUT_ONNX = "yolov4_fp8_qdq.onnx"


def dataloader():
    for p in glob.glob(os.path.join(CALIB_DIR, "*.jpg"))[:500]:
        img = cv2.imread(p)
        img = cv2.resize(img, (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        yield {"input": img}


class _SimpleDataReader:
    def __init__(self, gen):
        self._iter = iter(gen)

    def get_next(self):
        try:
            return next(self._iter)
        except StopIteration:
            return None


def _load_calib_batch():
    paths = glob.glob(os.path.join(CALIB_DIR, "*.jpg"))[:CALIB_COUNT]
    if not paths:
        raise RuntimeError(f"No calibration images found in {CALIB_DIR}")
    batches = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        batches.append(img)
    if not batches:
        raise RuntimeError("No valid calibration images could be read.")
    calib = np.stack(batches, axis=0)
    return {"input": calib}


def _make_quant_cfg():
    if hasattr(maq, "QuantizeConfig"):
        return maq.QuantizeConfig(format="fp8")
    if hasattr(maq, "QuantizationConfig"):
        return maq.QuantizationConfig(format="fp8")
    if hasattr(maq, "QuantConfig"):
        return maq.QuantConfig(format="fp8")
    return {"format": "fp8"}


def _run_quantize(onnx_path, data_gen):
    sig = inspect.signature(maq.quantize)
    kwargs = {}

    # Newer API: quantize(onnx_path=..., quantize_mode=..., calibration_data=..., output_path=...)
    if "onnx_path" in sig.parameters:
        if "quantize_mode" in sig.parameters:
            kwargs["quantize_mode"] = "fp8"
        elif "quant_mode" in sig.parameters:
            kwargs["quant_mode"] = "fp8"
        if "calibration_data" in sig.parameters:
            kwargs["calibration_data"] = _load_calib_batch()
        elif "calib_data" in sig.parameters:
            kwargs["calib_data"] = _load_calib_batch()
        if "output_path" in sig.parameters:
            kwargs["output_path"] = OUT_ONNX
        return maq.quantize(onnx_path=onnx_path, **kwargs)

    # Legacy API: quantize(model, quant_cfg=..., dataloader=...)
    model = onnx.load(onnx_path)

    quant_cfg = _make_quant_cfg()
    if "quant_cfg" in sig.parameters:
        kwargs["quant_cfg"] = quant_cfg
    elif "quant_config" in sig.parameters:
        kwargs["quant_config"] = quant_cfg
    elif "config" in sig.parameters:
        kwargs["config"] = quant_cfg

    if "dataloader" in sig.parameters:
        kwargs["dataloader"] = data_gen
    elif "calib_dataloader" in sig.parameters:
        kwargs["calib_dataloader"] = data_gen
    elif "data_loader" in sig.parameters:
        kwargs["data_loader"] = data_gen
    elif "calib_data_reader" in sig.parameters:
        kwargs["calib_data_reader"] = _SimpleDataReader(data_gen)
    elif "data_reader" in sig.parameters:
        kwargs["data_reader"] = _SimpleDataReader(data_gen)

    return maq.quantize(model, **kwargs)


ONNX_PATH = "yolov4_1_3_416_416_static.onnx"
try:
    quantized = _run_quantize(ONNX_PATH, dataloader())
except Exception:
    print("Quantize failed.")
    print("quantize signature:", inspect.signature(maq.quantize))
    print("Available attributes in modelopt.onnx.quantization:",
          [k for k in dir(maq) if "Quant" in k or "quant" in k])
    raise

if quantized is not None:
    onnx.save(quantized, OUT_ONNX)
print(f"saved {OUT_ONNX}")
