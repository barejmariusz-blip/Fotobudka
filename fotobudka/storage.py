"""Zapis stanu sesji (JSON w folderze) i plików XMP dla Lightrooma / Capture One / Bridge."""
from __future__ import annotations

import json
import os
import re

STATE_NAME = ".fotobudka.json"

LABELS = {0: None, 6: "Red", 7: "Yellow", 8: "Green", 9: "Blue"}


def load_state(folder: str) -> dict:
    p = os.path.join(folder, STATE_NAME)
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            if isinstance(d, dict):
                d.setdefault("photos", {})
                return d
    except (OSError, ValueError):
        pass
    return {"photos": {}}


def save_state(folder: str, state: dict) -> None:
    p = os.path.join(folder, STATE_NAME)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=0)
    os.replace(tmp, p)


def xmp_path(photo_path: str) -> str:
    return os.path.splitext(photo_path)[0] + ".xmp"


_TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Fotobudka by PanRodla">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmp:Rating="{rating}"{label}/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def _set_attr(text: str, name: str, value: str | None) -> str:
    """Ustawia/usuwa atrybut xmp:<name> w istniejącym XMP (atrybut lub element)."""
    attr_re = re.compile(r'\s+xmp:%s="[^"]*"' % name)
    elem_re = re.compile(r"<xmp:%s>[^<]*</xmp:%s>" % (name, name))
    if attr_re.search(text):
        return attr_re.sub("" if value is None else f' xmp:{name}="{value}"', text, count=1)
    if elem_re.search(text):
        return elem_re.sub("" if value is None else f"<xmp:{name}>{value}</xmp:{name}>", text, count=1)
    if value is None:
        return text
    m = re.search(r"<rdf:Description\b", text)
    if not m:
        return text
    ins = f' xmp:{name}="{value}"'
    if 'xmlns:xmp="' not in text:
        ins = ' xmlns:xmp="http://ns.adobe.com/xap/1.0/"' + ins
    return text[:m.end()] + ins + text[m.end():]


def write_xmp(photo_path: str, rating: int, rejected: bool, label: str | None) -> None:
    """rating 0-5; odrzucone = Rating -1 (standard Adobe)."""
    r = "-1" if rejected else str(rating)
    p = xmp_path(photo_path)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = _set_attr(text, "Rating", r)
        text = _set_attr(text, "Label", label)
    else:
        text = _TEMPLATE.format(rating=r, label=f'\n    xmp:Label="{label}"' if label else "")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
