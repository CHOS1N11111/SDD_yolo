import os
from pathlib import Path

import numpy as np
import pandas as pd


def read_image(path):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def load_reference_size(root, scene, video):
    ref_path = root / "annotations" / scene / video / "reference.jpg"
    if not ref_path.exists():
        return None
    buf = np.fromfile(ref_path, dtype=np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def class_to_id(name):
    if name == "Pedestrian":
        return 0
    if name == "Biker":
        return 1
    if name == "Skater":
        return 2
    if name == "Cart":
        return 3
    if name == "Car":
        return 4
    if name == "Bus":
        return 5
    return None


def main():
    cwd = Path.cwd()
    root = cwd if (cwd / "annotations").exists() else cwd.parent
    scene_filter = os.environ.get("SCENE")
    out_dir = root / "stanford_dd_yolo"
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total_labels = 0
    test_rows = []

    for ann in root.glob("annotations/*/*/annotations.txt"):
        scene = ann.parent.parent.name
        video = ann.parent.name
        if scene_filter and scene != scene_filter:
            continue
        frames_dir = root / "video" / scene / video / "frames"
        if not frames_dir.exists():
            continue
        if not any(frames_dir.glob("*.jpg")):
            # No frames found, skip this video.
            continue

        df = pd.read_csv(
            ann,
            names=["id", "left", "top", "right", "bottom", "frames", "a", "b", "c", "class"],
            sep=" ",
        )
        df.sort_values(["frames"], axis=0, ascending=True, inplace=True)
        selected = df[["left", "top", "right", "bottom", "frames", "class"]]

        ref_size = load_reference_size(root, scene, video)
        scale_x = 1.0
        scale_y = 1.0
        if ref_size:
            ref_w, ref_h = ref_size
            sample = next(frames_dir.glob("*.jpg"), None)
            if sample:
                img = read_image(str(sample))
                if img is not None:
                    frame_w = img.shape[1]
                    frame_h = img.shape[0]
                    if ref_w > 0 and ref_h > 0 and (ref_w != frame_w or ref_h != frame_h):
                        scale_x = frame_w / ref_w
                        scale_y = frame_h / ref_h

        frame_list = []
        for _, row in selected.iterrows():
            frame = int(row["frames"])
            if frame % 89 != 0:
                continue
            frame_name = f"{video}_{frame + 1}.jpg"
            frame_path = frames_dir / frame_name
            if not frame_path.exists():
                continue

            img = read_image(str(frame_path))
            if img is None:
                continue

            rows = selected.loc[selected["frames"] == frame]
            label_path = lbl_out / f"{video}_{frame + 1}.txt"
            lines = []
            for _, obj in rows.iterrows():
                cls_id = class_to_id(obj["class"])
                if cls_id is None:
                    continue
                left = float(obj["left"]) * scale_x
                top = float(obj["top"]) * scale_y
                right = float(obj["right"]) * scale_x
                bottom = float(obj["bottom"]) * scale_y
                x_norm = (left + (right - left) / 2.0) / img.shape[1]
                y_norm = (top + (bottom - top) / 2.0) / img.shape[0]
                w_norm = (right - left) / img.shape[1]
                h_norm = (bottom - top) / img.shape[0]
                lines.append(f"{cls_id} {x_norm} {y_norm} {w_norm} {h_norm}")
                total_labels += 1

            if not lines:
                continue
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            if frame not in frame_list:
                frame_list.append(frame)
                img_out_path = img_out / frame_name
                if not img_out_path.exists():
                    img_out_path.write_bytes(Path(frame_path).read_bytes())
                test_rows.append({"image": frame_name, "label": label_path.name})

    test_csv = out_dir / "test.csv"
    pd.DataFrame(test_rows).to_csv(test_csv, index=False)
    print(f"wrote {len(test_rows)} images, labels={total_labels}")


if __name__ == "__main__":
    import cv2

    raise SystemExit(main())
