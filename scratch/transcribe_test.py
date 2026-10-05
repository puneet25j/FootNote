"""
Checking Whiper model on clip
"""

import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

CLIP = Path("data/clips/lec01_first5min.wav")
OUT_DIR = Path("scratch/out")

def fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS.sss"""
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"

def main() -> None:
    model_size = sys.argv[1] if len(sys.argv) > 1 else "tiny"

    t0 = time.perf_counter()
    model = WhisperModel(model_size, device="cpu", compute_type="int8") 
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments, info = model.transcribe(str(CLIP), language="en", beam_size=5)
    segments = list(segments)
    transcribe_s = time.perf_counter() - t0

    for seg in segments:
        print(f"[{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}] {seg.text.strip()}")

    
    rtf = transcribe_s / info.duration
    print(
        f"\nmodel={model_size}  load={load_s:.1f}s  transcribe={transcribe_s:.1f}s  "
        f"audio={info.duration:.0f}s  RTF={rtf:.3f}  segments={len(segments)}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"lec01_5min_{model_size}.json"
    data = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    out_path.write_text(json.dumps(data, indent=2))
    print(f"saved {out_path}")
    
if __name__ == "__main__":
    main()