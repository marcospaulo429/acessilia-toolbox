"""Facade over the MinerU pipeline middle_json payload.

MinerU returns ``middle_json`` — a page-oriented tree where each page holds
``preproc_blocks`` (and ``discarded_blocks``) typed by :class:`BlockType`, plus
``page_size`` and ``page_idx``.  This facade exposes the same attribute-style
navigation the normalization builder expects from ``DoclingServeDocument``:
iterable collections of texts, tables, pictures and formulas with bbox and
page metadata hoisted onto each item.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

# Block types from mineru.utils.enum_class (inlined to avoid importing the
# mineru package here — the toolbox must stay free of ML runtimes).
TEXT_TYPES = ("text", "title", "list", "index")
TABLE_TYPES = ("table",)
IMAGE_TYPES = ("image",)
FORMULA_TYPES = ("interline_equation",)

_DISCARDABLE_KEYS = frozenset(
    {"preproc_blocks", "discarded_blocks", "page_idx", "page_size"}
)

class _BboxProxy:
    """Read-only [x0, y0, x1, y1] accessor over a block's bbox list.

    Property names follow the Docling bbox convention (``l/t/r/b``); the
    single-letter accessors carry a lint exemption because they match the
    existing normalization builder contract.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Sequence[float] | None) -> None:
        self._data = list(data) if data else []

    @property
    def l(self) -> float:  # noqa: E743 - matches Docling bbox naming
        return float(self._data[0]) if len(self._data) > 0 else 0.0

    @property
    def t(self) -> float:
        return float(self._data[1]) if len(self._data) > 1 else 0.0

    @property
    def r(self) -> float:
        return float(self._data[2]) if len(self._data) > 2 else 0.0

    @property
    def b(self) -> float:
        return float(self._data[3]) if len(self._data) > 3 else 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.l, self.t, self.r, self.b)


class _ItemProxy:
    """Uniform read access to one MinerU block regardless of nesting depth."""

    __slots__ = ("_block", "_page_idx", "_page_size")

    def __init__(
        self,
        block: Mapping[str, Any],
        page_idx: int,
        page_size: Sequence[float] | None,
    ) -> None:
        self._block = block
        self._page_idx = page_idx
        self._page_size = list(page_size or [])

    @property
    def label(self) -> str:
        return str(self._block.get("type", "unknown"))

    @property
    def page_no(self) -> int:
        return self._page_idx

    @property
    def bbox(self) -> _BboxProxy:
        return _BboxProxy(self._block.get("bbox"))

    @property
    def text(self) -> str:
        text = self._block_text()
        return str(text) if text is not None else ""

    def _block_text(self) -> Any:
        if "text" in self._block:
            return self._block["text"]
        lines = self._block.get("lines") or []
        parts: list[str] = []
        for line in lines:
            for span in line.get("spans", []):
                content = span.get("content") or span.get("text")
                if content:
                    parts.append(str(content))
        return "".join(parts) if parts else None

    @property
    def html(self) -> str | None:
        """Table body HTML, when the block carries one."""
        for sub in self._block.get("blocks", []) or []:
            if sub.get("type") in ("table_body",):
                for line in sub.get("lines", []) or []:
                    for span in line.get("spans", []) or []:
                        if span.get("type") == "table" and span.get("html"):
                            return str(span["html"])
        # Direct span fallback (flattened representations).
        for line in self._block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                if span.get("type") == "table" and span.get("html"):
                    return str(span["html"])
        return None

    @property
    def latex(self) -> str | None:
        """Formula LaTeX, when the block carries one."""
        if self._block.get("latex"):
            return str(self._block["latex"])
        for line in self._block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                content = span.get("content") or span.get("latex")
                equation_types = (
                    "equation",
                    "interline_equation",
                    "inline_equation",
                )
                if span.get("type") in equation_types and content:
                    return str(content)
        return None

    @property
    def raw(self) -> dict[str, Any]:
        return dict(self._block)


# MinerU block type -> Docling-style label understood by builder.LABEL_TO_TYPE
_BLOCK_LABEL = {
    "text": "text", "title": "section_header", "list": "list_item", "index": "text",
    "table": "table", "table_body": "table", "table_caption": "caption",
    "table_footnote": "footnote", "image": "picture", "image_body": "picture",
    "image_caption": "caption", "image_footnote": "footnote",
    "interline_equation": "formula", "code": "code", "code_body": "code",
    "code_caption": "caption", "header": "page_header", "footer": "page_footer",
    "page_number": "page_footer", "page_footnote": "footnote", "footnote": "footnote",
    "ref_text": "text", "aside_text": "text", "phonetic": "text",
}


class _Label:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value


class _CoordOrigin:
    value = "TOPLEFT"


class _ProvBbox(_BboxProxy):
    coord_origin = _CoordOrigin()


class _Prov:
    __slots__ = ("bbox", "charspan", "page_no")

    def __init__(self, page_no: int, bbox: Sequence[float] | None) -> None:
        self.page_no = page_no
        self.bbox = _ProvBbox(bbox)
        self.charspan = None


class _ReadingItem:
    """Flattened reading-order item with the attribute surface of a Docling ``DocItem``."""

    __slots__ = ("label", "level", "parent", "prov", "self_ref", "text")

    def __init__(
        self,
        label: str,
        text: str | None,
        page_no: int,
        bbox: Sequence[float] | None,
        ref: str,
        level: int | None = None,
    ) -> None:
        self.label = _Label(label)
        self.text = text
        self.prov = [_Prov(page_no, bbox)]
        self.self_ref = ref
        self.parent = None
        self.level = level


def _span_text(span: Mapping[str, Any]) -> str:
    kind = span.get("type")
    content = span.get("content") or span.get("latex") or ""
    if kind == "inline_equation" and content:
        return f"${content}$"
    if kind == "interline_equation" and content:
        return f"$${content}$$"
    if kind == "table" and span.get("html"):
        return str(span["html"])
    return str(content or "")


def _lines_text(block: Mapping[str, Any]) -> str | None:
    out = ""
    for line in block.get("lines") or []:
        parts = [_span_text(s) for s in line.get("spans") or []]
        piece = " ".join(p for p in parts if p).strip()
        if not piece:
            continue
        if not out:
            out = piece
        elif out.endswith("-") and piece[:1].islower():  # de-hyphenate line break
            out = out[:-1] + piece
        else:
            out += " " + piece
    return out or None


def _block_reading_text(block: Mapping[str, Any], block_type: str) -> str | None:
    if block_type in ("table", "table_body"):
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                if span.get("html"):
                    return str(span["html"])
        return _lines_text(block)
    if block_type == "interline_equation":
        latex = block.get("latex")
        if not latex:
            for line in block.get("lines") or []:
                for span in line.get("spans") or []:
                    if span.get("content"):
                        latex = span["content"]
                        break
                if latex:
                    break
        return f"$${latex}$$" if latex else None
    return block.get("text") or _lines_text(block)


class MineruDocument:
    """Facade over a MinerU ``middle_json`` payload."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = payload
        self._pages: list[Mapping[str, Any]] = list(
            payload.get("pdf_info") or []
        )

    # -- pages -----------------------------------------------------------

    @property
    def pages(self) -> list[dict[str, Any]]:
        return [
            {
                "page_no": page.get("page_idx", index),
                "width": (page.get("page_size") or [None, None])[0],
                "height": (page.get("page_size") or [None, None])[1],
            }
            for index, page in enumerate(self._pages)
        ]

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def num_pages(self) -> int:
        return len(self._pages)

    # -- Docling-compatible reading-order iteration -------------------------

    def iterate_items(self, **_: Any) -> Iterator[tuple[_ReadingItem, int]]:
        """Yield ``(item, level)`` in MinerU reading order, like ``DoclingDocument.iterate_items``.

        Container blocks (table/image/code) are flattened into their sub-blocks;
        ``discarded`` blocks are skipped. Page numbers are 1-based.
        """
        counter = 0

        def emit(
            block: Mapping[str, Any],
            page_no: int,
            parent_bbox: Sequence[float] | None = None,
        ) -> Iterator[tuple[_ReadingItem, int]]:
            nonlocal counter
            block_type = str(block.get("type", "text"))
            if block_type == "discarded":
                return
            bbox = block.get("bbox") or parent_bbox
            subs = block.get("blocks") or []
            if subs:
                for sub in subs:
                    yield from emit(sub, page_no, bbox)
                return
            label = _BLOCK_LABEL.get(block_type, "text")
            text = _block_reading_text(block, block_type)
            if not text and label != "picture":
                return
            counter += 1
            level = 1 if label == "section_header" else None
            ref = f"#/mineru/{counter}"
            yield _ReadingItem(label, text, page_no, bbox, ref, level), 1

        for page in self._pages:
            page_no = int(page.get("page_idx", 0)) + 1
            for block in page.get("preproc_blocks") or []:
                yield from emit(block, page_no)

    # -- block iteration ---------------------------------------------------

    def _blocks_of(
        self,
        types: tuple[str, ...],
        include_discarded: bool = False,
    ) -> Iterator[_ItemProxy]:
        for page in self._pages:
            page_idx = int(page.get("page_idx", 0))
            page_size = page.get("page_size") or []
            sources: list[Mapping[str, Any]] = list(
                page.get("preproc_blocks") or []
            )
            if include_discarded:
                sources += list(page.get("discarded_blocks") or [])
            for block in sources:
                if block.get("type") in types:
                    yield _ItemProxy(block, page_idx, page_size)

    @property
    def texts(self) -> list[_ItemProxy]:
        return list(self._blocks_of(TEXT_TYPES))

    @property
    def tables(self) -> list[_ItemProxy]:
        return list(self._blocks_of(TABLE_TYPES))

    @property
    def pictures(self) -> list[_ItemProxy]:
        return list(self._blocks_of(IMAGE_TYPES))

    @property
    def formulas(self) -> list[_ItemProxy]:
        return list(self._blocks_of(FORMULA_TYPES))

    # -- whole-document text ----------------------------------------------

    @property
    def full_text(self) -> str:
        parts: list[str] = []
        for page in self._pages:
            for block in page.get("preproc_blocks") or []:
                if block.get("type") == "discarded":
                    continue
                proxy = _ItemProxy(
                    block,
                    int(page.get("page_idx", 0)),
                    page.get("page_size"),
                )
                text = proxy.text
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    # -- metadata ------------------------------------------------------------

    @property
    def backend(self) -> str | None:
        return self._payload.get("_backend")

    @property
    def version_name(self) -> str | None:
        return self._payload.get("_version_name")

    @property
    def raw(self) -> dict[str, Any]:
        return dict(self._payload)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self._payload.items()
            if key not in _DISCARDABLE_KEYS
        }


__all__ = ["MineruDocument"]
