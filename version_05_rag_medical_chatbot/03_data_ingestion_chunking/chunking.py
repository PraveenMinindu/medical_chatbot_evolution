"""Document chunking for RAG pipelines.

WHY CHUNKING EXISTS:
    A document like medical_knowledge.md is too long to be indexed as one unit.
    If we embedded the entire document as a single vector, that vector would
    try to represent everything at once -- fever, headache, cough, cold,
    dehydration, prevention, warning signs -- all in one 384-dimensional point.

    When a user asks "what causes a headache?", the whole-document vector would
    score only moderately, because most of the document is NOT about headaches.
    The specific paragraph about headache causes would score much higher if it
    were stored separately.

    Chunking solves this by splitting the document into small, focused pieces.
    Each piece gets its own embedding. Each embedding represents only the meaning
    of that one chunk, not the entire document.

    The retrieval system then finds the specific chunk most relevant to the query,
    not the document as a whole.

WHY OVERLAP EXISTS:
    Imagine a document split at exactly sentence boundaries:

        Chunk 1: "...dehydration is a common trigger for headaches."
        Chunk 2: "When the body lacks fluid, brain tissue..."

    The first chunk ends with an important idea. The second chunk continues it.
    Without overlap, if a user asks about headache and dehydration, the retrieval
    system might return chunk 2 which starts in the middle of the explanation.

    With overlap, the end of chunk 1 is repeated at the start of chunk 2.
    This ensures that context is never lost at chunk boundaries. Every chunk
    contains some text from its neighbours, preserving continuity.

CHUNK SIZE TRADE-OFFS:
    Too small (e.g., 50 characters):
        - Each chunk lacks enough context to be meaningful on its own.
        - A sentence fragment cannot be understood without surrounding text.
        - The LLM receives many tiny pieces that don't add up to a coherent answer.

    Too large (e.g., 2000 characters):
        - The chunk covers too many topics at once.
        - The embedding vector is spread thin across many ideas.
        - Retrieval precision drops -- the wrong chunks get returned.

    Practical starting point: 300 to 600 characters per chunk with 10-20% overlap.
    This project uses 400 characters per chunk with 80 characters of overlap.
    These values can be adjusted in config.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """A single text chunk with full provenance metadata.

    Fields:
        chunk_id:    Unique identifier, e.g. "fever-chunk-003".
        text:        The actual text content of this chunk.
        topic:       The section heading this chunk belongs to.
        source_file: The filename the text was read from.
        char_start:  Character offset where this chunk begins in the source.
        char_end:    Character offset where this chunk ends in the source.
        chunk_index: Zero-based position of this chunk within its topic section.
    """
    chunk_id:    str
    text:        str
    topic:       str
    source_file: str
    char_start:  int
    char_end:    int
    chunk_index: int


def load_document(file_path: Path) -> str:
    """Read a markdown or text document and return its full content."""
    with file_path.open("r", encoding="utf-8") as f:
        return f.read()


def extract_sections(text: str) -> list[tuple[str, str]]:
    """Split a markdown document into (topic, content) pairs by H2 headings.

    Looks for lines starting with '## ' and treats each as a new section.
    Everything between two headings belongs to the first heading's topic.

    Args:
        text: Full document text.

    Returns:
        List of (topic_name, section_text) tuples.
        topic_name is the cleaned heading text.
        section_text is the paragraph text under that heading.

    Example:
        "## Fever\nFever is a temporary...\n## Headache\nA headache is..."
        ->
        [("fever", "Fever is a temporary..."),
         ("headache", "A headache is...")]
    """
    # Split on H2 headings (lines starting with '## ')
    pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        # No headings found -- treat the whole document as one section
        return [("general", text.strip())]

    sections = []
    for i, match in enumerate(matches):
        heading = match.group(1).strip()

        # Skip lines that look like metadata headers at the top of the doc
        if heading.startswith("General Educational") or heading.startswith("This document"):
            continue

        # Section content starts after the heading line
        content_start = match.end()
        # Section content ends at the next heading (or end of document)
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[content_start:content_end].strip()

        if section_text:
            # Use lowercase slug as topic identifier
            topic_slug = heading.lower().replace(" ", "_").replace("/", "_")
            sections.append((topic_slug, section_text))

    return sections


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """Split a text string into overlapping chunks of fixed character size.

    This is a character-based sliding window chunker. It is simple to understand
    and predictable in behaviour. Production systems often use token-based chunkers
    (splitting by number of tokens rather than characters) because LLM context
    windows are measured in tokens. For this learning project, characters are
    a clear and sufficient approximation.

    The chunker tries to split at sentence boundaries ('. ', '? ', '! ') within
    the last 60 characters of each chunk, to avoid cutting in the middle of a
    sentence. If no sentence boundary is found, it cuts at word boundaries instead.

    Args:
        text:       The section text to split into chunks.
        chunk_size: Target size of each chunk in characters.
        overlap:    Number of characters to repeat at the start of the next chunk.

    Returns:
        List of (chunk_text, char_start, char_end) tuples.

    Example with chunk_size=20, overlap=5:
        "The quick brown fox jumped over the lazy dog."
        -> [("The quick brown fox", 0, 19),
            ("n fox jumped over t", 14, 33),
            ...]
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to find a sentence boundary near the end of this chunk
        # to avoid cutting in the middle of a sentence
        if end < len(text):
            boundary = _find_sentence_boundary(text, end, lookback=60)
            if boundary > start:
                end = boundary

        chunk_content = text[start:end].strip()

        if len(chunk_content) > 30:
            # Only keep chunks with meaningful content (more than 30 characters)
            chunks.append((chunk_content, start, end))

        if end >= len(text):
            break

        # Next chunk starts 'overlap' characters before this chunk ended
        start = end - overlap

    return chunks


def _find_sentence_boundary(text: str, position: int, lookback: int) -> int:
    """Find the nearest sentence boundary before 'position'.

    Looks backwards from 'position' up to 'lookback' characters
    for a sentence-ending punctuation followed by whitespace.

    Returns:
        The position after the sentence boundary, or 'position' if none found.
    """
    search_start = max(0, position - lookback)
    segment = text[search_start:position]

    # Look for sentence-ending patterns from right to left
    for i in range(len(segment) - 1, -1, -1):
        if segment[i] in ".?!" and i + 1 < len(segment) and segment[i + 1] == " ":
            # Return position just after the punctuation and space
            return search_start + i + 2

    # No sentence boundary found -- try word boundary
    for i in range(len(segment) - 1, -1, -1):
        if segment[i] == " ":
            return search_start + i + 1

    return position


def build_chunks(
    file_path: Path,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Load a document, extract sections, and produce a flat list of Chunk objects.

    This is the main function called by the ingestion script.

    Args:
        file_path:  Path to the markdown or text source document.
        chunk_size: Target chunk size in characters.
        overlap:    Overlap size in characters between consecutive chunks.

    Returns:
        List of Chunk objects, each with unique chunk_id and metadata.
    """
    raw_text    = load_document(file_path)
    sections    = extract_sections(raw_text)
    source_file = file_path.name
    all_chunks: list[Chunk] = []

    for topic, section_text in sections:
        raw_chunks = chunk_text(section_text, chunk_size, overlap)

        for idx, (text, char_start, char_end) in enumerate(raw_chunks):
            chunk_id = f"{topic}-chunk-{idx:03d}"
            all_chunks.append(
                Chunk(
                    chunk_id    = chunk_id,
                    text        = text,
                    topic       = topic,
                    source_file = source_file,
                    char_start  = char_start,
                    char_end    = char_end,
                    chunk_index = idx,
                )
            )

    return all_chunks
