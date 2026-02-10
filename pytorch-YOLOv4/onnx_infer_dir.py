import argparse
import os
import sys

import numpy as np

try:
    import cv2
except Exception as exc:
    raise SystemExit("OpenCV is required. Install with: pip install opencv-python") from exc

try:
    import onnxruntime as ort
except Exception as exc:
    raise SystemExit("onnxruntime is required. Install with: pip install onnxruntime") from exc


ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(ROOT, "pytorch-YOLOv4")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from tool.utils import load_class_names, plot_boxes_cv2, post_processing


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
    return cv2.imread(path)


def _prepare_input(image_path, size):
    image_src = _imread_unicode(image_path)
    if image_src is None:
        raise ValueError(f"Failed to read image: {image_path}")
    in_w, in_h = size
    resized = cv2.resize(image_src, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    return image_src, img_in


def _list_images(input_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    names = []
    for name in os.listdir(input_dir):
        if os.path.splitext(name.lower())[1] in exts:
            names.append(name)
    names.sort()
    return [os.path.join(input_dir, n) for n in names]


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


def _pick_outputs(session, outputs):
    output_defs = session.get_outputs()
    name_to_arr = {d.name: arr for d, arr in zip(output_defs, outputs)}
    if "boxes" in name_to_arr and "confs" in name_to_arr:
        return name_to_arr["boxes"], name_to_arr["confs"]
    if len(outputs) >= 2:
        return outputs[0], outputs[1]
    raise RuntimeError(f"Unexpected outputs: {list(name_to_arr.keys())}")


def main():
    parser = argparse.ArgumentParser(description="ONNX forward inference on images or videos.")
    parser.add_argument("--onnx", required=True, help="Path to .onnx model")
    parser.add_argument("--input_dir", help="Directory of input images")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--video", help="Video file path")
    parser.add_argument("--names", required=True, help="Path to class names file")
    parser.add_argument("--out_dir", default="predictions_onnx", help="Output directory for images")
    parser.add_argument("--out", default="prediction_onnx.jpg", help="Output image path (single image)")
    parser.add_argument("--out_video", default="prediction_onnx.mp4", help="Output video path")
    parser.add_argument("--size", type=int, default=416, help="Input size (square). Default: 416")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.6, help="NMS threshold")
    parser.add_argument("--limit", type=int, default=10, help="Max images or frames to process")
    parser.add_argument("--use_gpu", action="store_true", help="Use CUDAExecutionProvider if available")
    args = parser.parse_args()

    args.onnx = _resolve_path(args.onnx)
    args.names = _resolve_path(args.names)
    args.input_dir = _resolve_path(args.input_dir)
    args.image = _resolve_path(args.image)
    args.video = _resolve_path(args.video)

    if not os.path.exists(args.onnx):
        raise SystemExit(f"ONNX not found: {args.onnx}")
    if not os.path.exists(args.names):
        raise SystemExit(f"Names file not found: {args.names}")
    if not args.input_dir and not args.image and not args.video:
        raise SystemExit("Provide one of --input_dir, --image, or --video.")
    if sum([bool(args.input_dir), bool(args.image), bool(args.video)]) != 1:
        raise SystemExit("Provide only one of --input_dir, --image, or --video.")
    if args.input_dir and not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input dir not found: {args.input_dir}")
    if args.image and not os.path.exists(args.image):
        raise SystemExit(f"Image not found: {args.image}")
    if args.video and not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    providers = ["CPUExecutionProvider"]
    if args.use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(args.onnx, providers=providers)
    input_name = session.get_inputs()[0].name
    class_names = load_class_names(args.names)
    processed = 0

    if args.input_dir:
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        images = _list_images(args.input_dir)
        if not images:
            raise SystemExit(f"No images found in: {args.input_dir}")

        for image_path in images:
            try:
                image_src, img_in = _prepare_input(image_path, (args.size, args.size))
            except ValueError as exc:
                print(str(exc))
                continue

            outputs = session.run(None, {input_name: img_in})
            boxes, confs = _pick_outputs(session, outputs)
            boxes_batch = post_processing(img_in, args.conf, args.nms, [boxes, confs])

            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(out_dir, f"{base}_onnx.jpg")
            plot_boxes_cv2(image_src, boxes_batch[0], savename=out_path, class_names=class_names)
            print(f"Saved: {out_path}")
            processed += 1
            if processed >= args.limit:
                break

        print(f"Processed {processed} image(s).")

    elif args.image:
        image_src, img_in = _prepare_input(args.image, (args.size, args.size))
        outputs = session.run(None, {input_name: img_in})
        boxes, confs = _pick_outputs(session, outputs)
        boxes_batch = post_processing(img_in, args.conf, args.nms, [boxes, confs])
        plot_boxes_cv2(image_src, boxes_batch[0], savename=args.out, class_names=class_names)
        print(f"Saved: {args.out}")

    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Failed to open video: {args.video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"Failed to open writer: {args.out_video}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_in = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
            img_in = np.expand_dims(img_in, axis=0)
            img_in /= 255.0
            img_in = np.ascontiguousarray(img_in)

            outputs = session.run(None, {input_name: img_in})
            boxes, confs = _pick_outputs(session, outputs)
            boxes_batch = post_processing(img_in, args.conf, args.nms, [boxes, confs])

            out_frame = plot_boxes_cv2(frame, boxes_batch[0], savename=None, class_names=class_names)
            writer.write(out_frame)
            processed += 1
            if processed >= args.limit:
                break

        cap.release()
        writer.release()
        print(f"Saved: {args.out_video}")
        print(f"Processed {processed} frame(s).")


if __name__ == "__main__":
    main()
