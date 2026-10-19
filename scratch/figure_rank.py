"""Which figure chunks compete for q25's figure seat, and where does lec01 p2 rank?"""
from footnote.index import make_where, search

Q = ("In the lecture 1 notes, what example array is drawn for the "
     "straightforward peak-finding algorithm, and where are its peaks?")

for rank, h in enumerate(search(Q, k=8, where=make_where(course="6.006", kind="figure")), 1):
    m = h["metadata"]
    print(f"{rank}. {h['score']:.3f} {m['lecture']} p{m['page_start']} | {h['text'][:80]!r}")