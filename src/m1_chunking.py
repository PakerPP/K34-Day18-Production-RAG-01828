from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

_SEMANTIC_ENCODER = None


def _get_semantic_encoder():
    """Load all-MiniLM-L6-v2 một lần rồi cache — encode lại model mỗi call rất chậm."""
    global _SEMANTIC_ENCODER
    if _SEMANTIC_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_ENCODER


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0],
                      metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    embeddings = _get_semantic_encoder().encode(sentences)

    def cosine_sim(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    chunks: list[Chunk] = []
    group = [sentences[0]]
    for i in range(1, len(sentences)):
        # Câu i khác chủ đề với câu i-1 → đóng chunk hiện tại, mở chunk mới.
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            chunks.append(Chunk(
                text=" ".join(group).strip(),
                metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)},
            ))
            group = []
        group.append(sentences[i])

    if group:
        chunks.append(Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)},
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _split_by_size(text: str, max_size: int) -> list[str]:
    """Cắt text thành các đoạn ≤ max_size, ưu tiên ranh giới câu rồi mới tới từ."""
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n", text) if u.strip()]
    pieces: list[str] = []
    current = ""

    for unit in units:
        # Đơn vị dài hơn max_size → buộc phải cắt theo từ.
        while len(unit) > max_size:
            head, unit = unit[:max_size].rsplit(" ", 1) if " " in unit[:max_size] else (unit[:max_size], unit[max_size:])
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.append(head.strip())
            unit = unit.strip()
        if current and len(current) + len(unit) + 1 > max_size:
            pieces.append(current.strip())
            current = ""
        current = f"{current} {unit}".strip()

    if current.strip():
        pieces.append(current.strip())
    return [p for p in pieces if p]


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    # --- Bước 1: gộp paragraphs thành parents (mỗi parent ≤ parent_size chars) ---
    parents: list[Chunk] = []

    # parent_id phải unique TOÀN CORPUS: pipeline gọi hàm này riêng cho từng
    # document, nếu chỉ đánh số "parent_0" thì mọi document đều có parent_0 →
    # child của file A trỏ nhầm sang parent của file B.
    doc_key = str(metadata.get("source", "doc"))

    def _add_parent(body: str) -> None:
        pid = f"{doc_key}::parent_{len(parents)}"
        parents.append(Chunk(
            text=body.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                      "strategy": "hierarchical"},
        ))

    current = ""
    for para in paragraphs:
        # Paragraph dài hơn parent_size → tự nó phải được cắt nhỏ, nếu không
        # parent sẽ vượt giới hạn (VD 1 paragraph 5000 chars → parent 5000 chars).
        if len(para) > parent_size:
            if current.strip():
                _add_parent(current)
                current = ""
            for piece in _split_by_size(para, parent_size):
                _add_parent(piece)
            continue
        if current and len(current) + len(para) > parent_size:
            _add_parent(current)
            current = ""
        current += para + "\n\n"
    if current.strip():
        _add_parent(current)

    # --- Bước 2: mỗi parent → children (mỗi child ≤ child_size chars) ---
    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        for child_text in _split_by_size(parent.text, child_size):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "strategy": "hierarchical",
                          "chunk_index": len(children)},
                parent_id=pid,
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}

    # Capturing group giữ lại chính header trong kết quả split.
    sections = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    header = ""
    content = ""

    def _flush() -> None:
        body = content.strip()
        if not header and not body:
            return
        full = f"{header}\n{body}".strip() if header else body
        chunks.append(Chunk(
            text=full,
            metadata={**metadata, "section": header.lstrip("# ").strip(),
                      "strategy": "structure", "chunk_index": len(chunks)},
        ))

    for part in sections:
        if not part.strip():
            continue
        if re.match(r"^#{1,3}\s+", part):
            _flush()          # đóng section trước đó
            header = part.strip()
            content = ""
        else:
            content += part

    _flush()                  # section cuối cùng

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
