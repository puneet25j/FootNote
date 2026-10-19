"""Why is q25 refused? Show what the model sees, then try two questions on the SAME sources."""
from footnote import llm
from footnote.qa import build_prompt, retrieve, system_prompt

REAL = ("In the lecture 1 notes, what example array is drawn for the "
        "straightforward peak-finding algorithm, and where are its peaks?")
NO_PEAKS = ("In the lecture 1 notes, what example array is drawn for the "
            "straightforward peak-finding algorithm?")

hits = retrieve(REAL, "6.006")          # retrieve once, reuse for both questions
print("SOURCES THE MODEL GETS:")
for n, h in enumerate(hits, 1):
    m = h["metadata"]
    print(f"[{n}] {h['score']:.3f} {m['lecture']} {m['source_type']} {m.get('kind')} | {h['text'][:100]!r}")

for name, q in [("REAL", REAL), ("NO_PEAKS", NO_PEAKS)]:
    print(f"\n--- {name} ---")
    print(llm.generate(build_prompt(q, hits), system=system_prompt("6.006")))