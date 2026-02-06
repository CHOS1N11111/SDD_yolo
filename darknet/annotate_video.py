import argparse
import re
import shutil
import subprocess
from pathlib import Path


FRAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<num>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Darknet on video frames and assemble an output video."
    )
    parser.add_argument("--scene", required=True, help="Scene name, e.g. bookstore")
    parser.add_argument("--video", required=True, help="Video folder name, e.g. video0")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Dataset root (default: parent of darknet folder)",
    )
    parser.add_argument(
        "--frames-dir",
        default="",
        help="Override frames directory (default: <root>/video/<scene>/<video>/frames)",
    )
    parser.add_argument(
        "--weights",
        default="backup/yolov4-custom_last.weights",
        help="Weights file path (relative to darknet or absolute)",
    )
    parser.add_argument(
        "--cfg",
        default="cfg/yolov4-custom.cfg",
        help="Cfg file path (relative to darknet or absolute)",
    )
    parser.add_argument(
        "--data",
        default="data/obj.data",
        help="Data file path (relative to darknet or absolute)",
    )
    parser.add_argument(
        "--darknet",
        default="build_cuda/Release/darknet.exe",
        help="Darknet executable (relative to darknet or absolute)",
    )
    parser.add_argument("--thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--framerate", type=float, default=30, help="Output video FPS")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output frames directory (default: <root>/results/<scene>_<video>_pred_frames)",
    )
    parser.add_argument(
        "--out-video",
        default="",
        help="Output video path (default: <root>/results/<scene>_<video>_pred.mp4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of frames (0 = all)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip first N frames after sorting",
    )
    return parser.parse_args()


def numeric_suffix(name: str) -> int:
    m = FRAME_RE.match(name)
    if not m:
        return -1
    return int(m.group("num"))


def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def main() -> int:
    args = parse_args()
    darknet_dir = Path(__file__).resolve().parent
    root = Path(args.root)

    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    else:
        frames_dir = root / "video" / args.scene / args.video / "frames"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = root / "results" / f"{args.scene}_{args.video}_pred_frames"

    if args.out_video:
        out_video = Path(args.out_video)
    else:
        out_video = root / "results" / f"{args.scene}_{args.video}_pred.mp4"

    darknet_exe = resolve_path(args.darknet, darknet_dir)
    cfg = resolve_path(args.cfg, darknet_dir)
    data = resolve_path(args.data, darknet_dir)
    weights = resolve_path(args.weights, darknet_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not darknet_exe.exists():
        raise FileNotFoundError(f"Darknet executable not found: {darknet_exe}")
    if not cfg.exists():
        raise FileNotFoundError(f"Cfg file not found: {cfg}")
    if not data.exists():
        raise FileNotFoundError(f"Data file not found: {data}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(
        (p for p in frames_dir.glob(f"{args.video}_*.jpg")),
        key=lambda p: numeric_suffix(p.stem),
    )
    if args.start:
        frames = frames[args.start :]
    if args.limit and args.limit > 0:
        frames = frames[: args.limit]

    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir} with prefix {args.video}_")

    pred_path = darknet_dir / "predictions.jpg"

    print(f"Frames: {len(frames)}")
    print(f"Output frames: {out_dir}")
    for idx, frame in enumerate(frames, start=1):
        cmd = [
            str(darknet_exe),
            "detector",
            "test",
            str(data),
            str(cfg),
            str(weights),
            str(frame),
            "-dont_show",
            "-thresh",
            str(args.thresh),
        ]
        subprocess.run(cmd, cwd=darknet_dir, check=True)
        if not pred_path.exists():
            raise RuntimeError("predictions.jpg not found after darknet run.")

        num = numeric_suffix(frame.stem)
        if num < 0:
            raise RuntimeError(f"Unexpected frame name: {frame.name}")
        out_name = f"{args.video}_{num:06d}.jpg"
        shutil.copy2(pred_path, out_dir / out_name)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(frames)} frames...")

    # Assemble video
    input_pattern = str(out_dir / f"{args.video}_%06d.jpg")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.framerate),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_video),
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    print(f"Done. Output video: {out_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
