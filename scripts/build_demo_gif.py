"""One-off: stitch annotated keyframes into a looping demo GIF for the README."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    candidates = sorted(glob.glob("data/keyframes/*/*.jpg"))
    if not candidates:
        print("no keyframes found; run `make verify` first", file=sys.stderr)
        return 1

    # Pick a narrative arc: the 5 interaction peaks in chronological order,
    # plus the late-video motion transition (the technician steps back).
    pick_frames = [14, 41, 114, 119, 123, 141, 163]
    by_frame: dict[int, str] = {}
    for path in candidates:
        name = Path(path).name
        for f in pick_frames:
            if f"_{f}.jpg" in name and ("interaction" in name or "obj2_motion" in name):
                by_frame.setdefault(f, path)

    selected = [by_frame[f] for f in pick_frames if f in by_frame]
    if not selected:
        print("no matching keyframes; using all interaction peaks", file=sys.stderr)
        selected = sorted([p for p in candidates if "interaction" in p])
    print(f"using {len(selected)} frames: {[Path(p).name for p in selected]}")

    target_w = 800
    images = []
    for p in selected:
        im = Image.open(p).convert("RGB")
        h = int(im.height * target_w / im.width)
        images.append(im.resize((target_w, h), Image.LANCZOS))

    out = Path("docs/demo.gif")
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=900,
        loop=0,
        optimize=True,
    )
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
