"""Check highlight lengths (<=80 characters incl. spaces), abstract length and citation rules."""
import re
from pathlib import Path

s = Path(__file__).with_name("Causal_WeatherGraph_revised.tex").read_text(encoding="utf-8")
hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", s, re.S).group(1)
for item in re.findall(r"\\item (.*)", hl):
    txt = item.replace("--", "\u2013").replace("\\%", "%").strip()
    print(f"{len(txt):3d}  {txt}")
ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S).group(1)
ab = re.sub(r"\$[^$]*\$", "X", ab)
print("abstract words:", len(ab.split()))
grouped = re.findall(r"\\cite[pt]?\{[^}]*,[^}]*\}", s)
print("grouped citation commands:", grouped)
keys = re.findall(r"\\cite[pt]?\{([^}]*)\}", s)
order = []
for k in keys:
    if k not in order:
        order.append(k)
print("unique cited:", len(order))
bib = Path(__file__).with_name("Causal_WeatherGraph_references.bib").read_text(encoding="utf-8")
bibkeys = re.findall(r"@\w+\{([^,]+),", bib)
print("bib entries:", len(bibkeys), "uncited:", [k for k in bibkeys if k not in order])
journals = re.findall(r"(?:journal|booktitle|publisher)\s*=\s*\{([^}]*)\}", bib)
from collections import Counter
print("max per venue:", Counter(journals).most_common(3))
authors = re.findall(r"author\s*=\s*\{([^}]*)\}", bib)
names = Counter()
for a in authors:
    for n in a.split(" and "):
        names[n.split(",")[0].strip()] += 1
print("max per author surname:", names.most_common(5))
