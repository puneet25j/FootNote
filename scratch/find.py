"""Find where a phrase is said or written, to label eval questions.

Usage: python scratch/find.py 6.006 "pointer machine"
"""

import json
import re
import sys
from pathlib import Path

course, pattern = sys.argv[1], sys.argv[2]
for path in sorted(Path(f"data/courses/{course}/processed").glob("*/*.json")):
    doc = json.loads(path.read_text())
    for seg in doc["segments"]:
        if re.search(pattern, seg["text"],  re.IGNORECASE):
            if doc["source_type"] == "video":
                m, s = divmod(int(seg["start"]), 60)
                where = f"{int(seg['start'])}s ({m}:{s:02d})"
            else:
                where = f"p{seg['page']}"
            print(f"{doc['lecture']} {doc['source_type']:5s} {where:>14}  {seg['text'][:80]}")