"""Lecture de la sortie texte de SAM 3.1 (Meta Model API).

Format documenté (dev.meta.ai/docs/sam/reading-segmentation), pour une image :

    <0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561|><|mask;x=0;y=0;data=46,101,~...|>,1<|box;...|>...

- un enregistrement par objet : un id, puis une boîte, puis un masque ;
- `c=<confiance>` apparaît si on demande `metadata.include_confidence = "true"`.

On ne décode pas les masques ici (encodage base85 propre à Meta) : pour l'overlay, les boîtes suffisent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Les champs d'une boîte ne contiennent que des chiffres, lettres, '=', ';' et '.', donc [^|]* est sûr.
# On ne cherche pas la fin des masques avec une regex : leur charge utile base85 peut contenir '|'.
_BOX_RE = re.compile(r"(\d+)<\|box;([^|]*)\|>")


@dataclass(frozen=True)
class Box:
    obj_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    frame_w: int
    frame_h: int
    confidence: Optional[float] = None
    has_mask: bool = False

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    def scaled(self, img_w: int, img_h: int) -> "Box":
        """Remet la boîte à l'échelle de l'image affichée (si SAM a travaillé sur une autre taille)."""
        if not self.frame_w or not self.frame_h or (img_w == self.frame_w and img_h == self.frame_h):
            return self
        sx, sy = img_w / self.frame_w, img_h / self.frame_h
        return Box(
            obj_id=self.obj_id,
            x1=round(self.x1 * sx), y1=round(self.y1 * sy),
            x2=round(self.x2 * sx), y2=round(self.y2 * sy),
            frame_w=img_w, frame_h=img_h,
            confidence=self.confidence, has_mask=self.has_mask,
        )


def _kv(fields: str) -> dict:
    out = {}
    for part in fields.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_boxes(output_text: str) -> List[Box]:
    """Extrait les boîtes d'une sortie SAM 3.1 sur image. Renvoie [] si aucun objet trouvé."""
    if not output_text:
        return []
    boxes: List[Box] = []
    for m in _BOX_RE.finditer(output_text):
        obj_id, box_fields = m.group(1), m.group(2)
        has_mask = output_text.startswith("<|mask", m.end())
        kv = _kv(box_fields)
        try:
            boxes.append(Box(
                obj_id=int(obj_id),
                x1=int(kv["x1"]), y1=int(kv["y1"]), x2=int(kv["x2"]), y2=int(kv["y2"]),
                frame_w=int(kv.get("w", 0)), frame_h=int(kv.get("h", 0)),
                confidence=float(kv["c"]) if "c" in kv else None,
                has_mask=has_mask,
            ))
        except (KeyError, ValueError):
            continue
    return boxes
