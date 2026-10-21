"""Rank figure chunks for one eval question:  python -m scratch.figure_rank q24"""
import json
import sys

from footnote.index import make_where, search

qid = sys.argv[1] if len(sys.argv) > 1 else "q25"
questions = [json.loads(line) for line in open("eval/questions.jsonl") if line.strip()]
q = next(q for q in questions if q["id"] == qid)
gold = {(g["lecture"], g.get("page")) for g in q["gold"]}

print(q["question"])

for rank, h in enumerate(search(q["question"], k=8, where=make_where(course=q["course"], kind="figure")), 1):
    m = h["metadata"]
    mark = "   <-- gold page" if (m["lecture"], m["page_start"]) in gold else ""
    print(f"{rank}. {h['score']:.3f} {m['lecture']} p{m['page_start']} | {h['text'][:70]!r}{mark}")