"""Document chunking for ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    text: str
    topic: str
    source_file: str
    char_start: int
    char_end: int
    chunk_index: int


def load_document(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def extract_sections(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [("general", text.strip())]

    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        if heading.startswith("General Educational") or heading.startswith("This document"):
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            topic = heading.lower().replace(" ", "_").replace("/", "_")
            sections.append((topic, body))
    return sections


def _find_boundary(text: str, position: int, lookback: int = 60) -> int:
    search_start = max(0, position - lookback)
    segment = text[search_start:position]
    for i in range(len(segment) - 1, -1, -1):
        if segment[i] in ".?!" and i + 1 < len(segment) and segment[i + 1] == " ":
            return search_start + i + 2
    for i in range(len(segment) - 1, -1, -1):
        if segment[i] == " ":
            return search_start + i + 1
    return position


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    if not text:
        return []
    out: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = _find_boundary(text, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            out.append((chunk, start, end))
        if end >= len(text):
            break
        start = end - overlap
    return out


def build_chunks(file_path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    text = load_document(file_path)
    sections = extract_sections(text)
    source = file_path.name
    all_chunks: list[Chunk] = []
    for topic, section_text in sections:
        for idx, (chunk_text_value, char_start, char_end) in enumerate(chunk_text(section_text, chunk_size, overlap)):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{topic}-chunk-{idx:03d}",
                    text=chunk_text_value,
                    topic=topic,
                    source_file=source,
                    char_start=char_start,
                    char_end=char_end,
                    chunk_index=idx,
                )
            )
    return all_chunks

