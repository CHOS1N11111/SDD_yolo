import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

try:
    import cv2
except Exception as exc:
    raise SystemExit("OpenCV is required. Install with: pip install opencv-python") from exc

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(ROOT, "pytorch-YOLOv4")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from tool.utils import load_class_names, plot_boxes_cv2, post_processing


def _parse_dims(dim_str):
    # "1x10647x1x4" -> [1, 10647, 1, 4]
    return [int(x) for x in dim_str.split("x") if x]


def _load_trt_outputs(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    outputs = {}
    for item in data:
        name = item["name"]
        dims = _parse_dims(item["dimensions"])
        values = np.array(item["values"], dtype=np.float32)
        outputs[name] = values.reshape(dims)
    return outputs


def _load_trt_outputs_retry(json_path, retries=3, delay=0.2):
    last_err = None
    for _ in range(retries):
        try:
            return _load_trt_outputs(json_path)
        except json.JSONDecodeError as err:
            last_err = err
            time.sleep(delay)
    raise last_err


def _imread_unicode(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    except Exception:
        pass
    # Fallback for unusual cases
    return cv2.imread(path)


def _prepare_input(image_path, size):
    image_src = _imread_unicode(image_path)
    if image_src is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    in_w, in_h = size
    resized = cv2.resize(image_src, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    return image_src, img_in


def _run_trt(engine_path, input_bin, output_json, shape_str, trt_exe):
    cmd = [
        trt_exe,
        f"--loadEngine={engine_path}",
        f"--loadInputs=input:{input_bin}",
        f"--exportOutput={output_json}",
        f"--shapes=input:{shape_str}",
        "--iterations=1",
        "--warmUp=0",
        "--duration=1",
    ]
    subprocess.run(cmd, check=True)


def _resolve_path(path):
    if path is None:
        return None
    if os.path.exists(path):
        return path
    cand = os.path.join(ROOT, path)
    if os.path.exists(cand):
        return cand
    parent = os.path.dirname(ROOT)
    cand = os.path.join(parent, path)
    if os.path.exists(cand):
        return cand
    return path


def _list_images(input_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    names = []
    for name in os.listdir(input_dir):
        if os.path.splitext(name.lower())[1] in exts:
            names.append(name)
    names.sort()
    return [os.path.join(input_dir, n) for n in names]

def _ensure_tmp_root():
    root = os.path.join(os.getcwd(), "_trt_tmp")
    os.makedirs(root, exist_ok=True)
    return root


def main():
    parser = argparse.ArgumentParser(description="Verify TensorRT-RTX engine with one image.")
    parser.add_argument("--engine", required=True, help="Path to .trt engine file")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--input_dir", help="Directory of input images")
    parser.add_argument("--names", required=True, help="Path to class names file")
    parser.add_argument("--size", type=int, default=416, help="Input size (square). Default: 416")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.6, help="NMS threshold")
    parser.add_argument("--trt", default="tensorrt_rtx.exe", help="Path to tensorrt_rtx executable")
    parser.add_argument("--out", default="predictions_trt_rtx.jpg", help="Output image path (single image)")
    parser.add_argument("--out_dir", default="predictions_trt_rtx", help="Output directory (batch)")
    parser.add_argument("--limit", type=int, default=10, help="Max images to process from input_dir")
    parser.add_argument("--keep", action="store_true", help="Keep intermediate input/output files")
    args = parser.parse_args()

    trt_exe = args.trt
    if os.path.sep not in trt_exe and shutil.which(trt_exe) is None:
        raise SystemExit("tensorrt_rtx not found. Pass --trt <full_path_to_tensorrt_rtx.exe> or add it to PATH.")

    args.engine = _resolve_path(args.engine)
    args.names = _resolve_path(args.names)
    args.image = _resolve_path(args.image)
    if args.input_dir:
        args.input_dir = _resolve_path(args.input_dir)

    if args.engine and not os.path.exists(args.engine):
        raise SystemExit(f"Engine not found: {args.engine}")
    if args.names and not os.path.exists(args.names):
        raise SystemExit(f"Names file not found: {args.names}")

    if not args.image and not args.input_dir:
        raise SystemExit("Provide --image or --input_dir.")
    if args.image and args.input_dir:
        raise SystemExit("Provide only one of --image or --input_dir.")

    if args.input_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        images = _list_images(args.input_dir)
        if not images:
            raise SystemExit(f"No images found in: {args.input_dir}")
    else:
        images = [args.image]

    processed = 0
    tmp_root = _ensure_tmp_root()
    for image_path in images:
        try:
            image_src, img_in = _prepare_input(image_path, (args.size, args.size))
        except SystemExit as exc:
            print(str(exc))
            continue

        with tempfile.TemporaryDirectory(dir=tmp_root) as tmpdir:
            input_bin = os.path.join(tmpdir, "input.bin")
            output_json = os.path.join(tmpdir, "output.json")
            img_in.tofile(input_bin)

            input_arg = os.path.relpath(input_bin, os.getcwd())
            output_arg = os.path.relpath(output_json, os.getcwd())
            shape_str = f"1x3x{args.size}x{args.size}"
            _run_trt(args.engine, input_arg, output_arg, shape_str, trt_exe)

            try:
                outputs = _load_trt_outputs_retry(output_json)
            except json.JSONDecodeError as exc:
                print(f"Failed to parse output JSON, skipping: {output_json}")
                print(f"Reason: {exc}")
                continue
            if "boxes" not in outputs or "confs" not in outputs:
                raise SystemExit(f"Missing outputs in JSON. Got keys: {list(outputs.keys())}")

            boxes = outputs["boxes"]
            confs = outputs["confs"]
            boxes_batch = post_processing(img_in, args.conf, args.nms, [boxes, confs])

            class_names = load_class_names(args.names)
            if args.input_dir:
                base = os.path.splitext(os.path.basename(image_path))[0]
                out_path = os.path.join(args.out_dir, f"{base}_trt_rtx.jpg")
            else:
                out_path = args.out
            plot_boxes_cv2(image_src, boxes_batch[0], savename=out_path, class_names=class_names)
            print(f"Saved: {out_path}")
            processed += 1

            if args.keep:
                keep_dir = os.path.splitext(out_path)[0] + "_debug"
                os.makedirs(keep_dir, exist_ok=True)
                shutil.copy2(input_bin, os.path.join(keep_dir, "input.bin"))
                shutil.copy2(output_json, os.path.join(keep_dir, "output.json"))
                print(f"Saved intermediates to: {keep_dir}")

        if args.input_dir and processed >= args.limit:
            break

    if args.input_dir:
        print(f"Processed {processed} image(s).")


if __name__ == "__main__":
    main()
