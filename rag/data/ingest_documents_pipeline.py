import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import fitz
import torch
from FlagEmbedding import BGEM3FlagModel
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from write_embeddings_to_postgres import (
    get_postgres_connection,
    insert_chunks_with_embeddings,
    normalize_metadata,
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


@dataclass
class IngestionJob:
    path: Path
    metadata: Dict[str, Any]


def collect_jobs_from_inputs(input_paths: List[str], base_metadata: Dict[str, Any]) -> List[IngestionJob]:
    jobs: List[IngestionJob] = []
    for input_path in input_paths:
        path = Path(input_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    jobs.append(IngestionJob(path=file_path, metadata=dict(base_metadata)))
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            jobs.append(IngestionJob(path=path, metadata=dict(base_metadata)))

    if not jobs:
        raise ValueError(
            "No compatible documents found. Supported formats: .pdf, .txt, .md, .markdown."
        )
    return jobs


def collect_jobs_from_manifest(manifest_path: str) -> List[IngestionJob]:
    path = Path(manifest_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Manifest must be a JSON list of objects.")

    jobs: List[IngestionJob] = []
    for item in payload:
        file_path = Path(item["path"]).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {file_path}")
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        jobs.append(IngestionJob(path=file_path, metadata=dict(item.get("metadata", {}))))

    if not jobs:
        raise ValueError("Manifest does not contain any supported documents.")
    return jobs


def read_pdf_pages(path: Path) -> List[Dict[str, Any]]:
    extracted_pages: List[Dict[str, Any]] = []
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue
            extracted_pages.append({"text": page_text, "metadata": {"page_number": page_index}})
    return extracted_pages


def read_markdown_sections(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "title_1"), ("###", "title_2")],
        strip_headers=False,
    )
    docs = splitter.split_text(content)

    if not docs:
        return [{"text": content, "metadata": {}}]

    sections: List[Dict[str, Any]] = []
    for doc in docs:
        section_name = " > ".join(
            [value for key, value in doc.metadata.items() if key in ("title_1", "title_2") and value]
        )
        sections.append(
            {
                "text": doc.page_content.strip(),
                "metadata": {"section": section_name or "unknown"},
            }
        )
    return sections


def read_plain_text(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    return [{"text": content, "metadata": {}}]


def extract_document_units(job: IngestionJob) -> List[Dict[str, Any]]:
    extension = job.path.suffix.lower()
    if extension == ".pdf":
        return read_pdf_pages(job.path)
    if extension in {".md", ".markdown"}:
        return read_markdown_sections(job.path)
    if extension == ".txt":
        return read_plain_text(job.path)
    return []


def build_chunks_for_job(
    job: IngestionJob,
    chunk_size: int,
    chunk_overlap: int,
    language: str,
    embedding_model: str,
) -> List[Dict[str, Any]]:
    units = extract_document_units(job)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    raw_chunks: List[Dict[str, Any]] = []
    for unit in units:
        text = unit["text"].strip()
        if not text:
            continue
        split_texts = splitter.split_text(text)
        for split in split_texts:
            metadata = dict(job.metadata)
            metadata.update(unit.get("metadata", {}))
            metadata["source_file"] = job.path.name
            metadata["source_path"] = str(job.path)
            metadata["format"] = job.path.suffix.lstrip(".").lower()
            metadata["language"] = language
            metadata["embedding_model"] = embedding_model
            raw_chunks.append({"page_content": split, "metadata": metadata})

    chunk_total = len(raw_chunks)
    for index, chunk in enumerate(raw_chunks):
        chunk["metadata"]["chunk_index"] = index
        chunk["metadata"]["chunk_total"] = chunk_total
        chunk["metadata"] = normalize_metadata(chunk["metadata"])

    return raw_chunks


def enrich_chunks_with_embeddings(
    chunks: List[Dict[str, Any]],
    embedding_model: str,
    max_tokens: int,
    batch_size: int,
) -> None:
    if not chunks:
        return

    model = BGEM3FlagModel(
        embedding_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=True,
    )

    texts = [chunk["page_content"] for chunk in chunks]
    vectors = model.encode(
        texts,
        max_length=max_tokens,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]

    for index, vector in enumerate(vectors):
        chunks[index]["embedding"] = vector.tolist()


def save_local_output(chunks: List[Dict[str, Any]], output_json: str) -> None:
    output_path = Path(output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic RAG ingestion pipeline for elevator maintenance documents."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help=(
            "Document files or folders to ingest. "
            "Example: --inputs ./documents/new_manuals ./documents/troubleshooting.txt"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest (list of {path, metadata}) for per-document metadata.",
    )
    parser.add_argument("--env-file", default=None, help="Path to vecdb env file.")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path for generated chunks.")
    parser.add_argument("--brand", default="unknown", help="Elevator brand metadata.")
    parser.add_argument("--elevator-model", default="unknown", help="Elevator model/type metadata.")
    parser.add_argument("--document-type", default="maintenance_manual", help="Document type metadata.")
    parser.add_argument("--document-version", default="unknown", help="Document version metadata.")
    parser.add_argument("--language", default="english", help="Document language metadata.")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3", help="Embedding model name.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Embedding model max sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not args.inputs and not args.manifest:
        raise ValueError("Provide --inputs and/or --manifest.")

    base_metadata = {
        "brand": args.brand,
        "elevator_model": args.elevator_model,
        "document_type": args.document_type,
        "document_version": args.document_version,
    }

    jobs: List[IngestionJob] = []
    if args.inputs:
        jobs.extend(collect_jobs_from_inputs(args.inputs, base_metadata))
    if args.manifest:
        jobs.extend(collect_jobs_from_manifest(args.manifest))

    chunks: List[Dict[str, Any]] = []
    for job in jobs:
        chunks.extend(
            build_chunks_for_job(
                job=job,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                language=args.language,
                embedding_model=args.embedding_model,
            )
        )

    if not chunks:
        raise ValueError("No chunks were generated from input documents.")

    enrich_chunks_with_embeddings(
        chunks=chunks,
        embedding_model=args.embedding_model,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )

    if args.output_json:
        save_local_output(chunks, args.output_json)

    conn = get_postgres_connection(args.env_file)
    try:
        inserted = insert_chunks_with_embeddings(conn, chunks)
    finally:
        conn.close()

    print(f"Ingestion completed. Inserted {inserted} chunks into vector database.")


if __name__ == "__main__":
    # Example usage:
    # python ingest_documents_pipeline.py --inputs ./documents/manual.pdf --brand Hyundai --elevator-model HDX
    main()
