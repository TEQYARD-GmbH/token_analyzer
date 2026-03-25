# /// script
# dependencies = [
#     "PyMuPDF",
#     "beautifulsoup4",
#     "chardet",
#     "cyclopts",
#     "docling-parse",
#     "docling>=2.0.0",
#     "llama-index",
#     "llama-index-readers-file",
#     "llama-index-readers-json>=0.4.1",
#     "llama-index-readers-web",
#     "llama-index-readers-docling",
#     "llama-index-node-parser-docling",
#     "markitdown[docx,pptx,pdf]",
#     "openpyxl",
#     "pandas",
#     "pikepdf",
#     "pypdfium2",
#     "libpff-python",
#     "python-docx",
#     "tiktoken",
#     "xlrd",
# ]
# ///

import csv
import glob
import logging
import multiprocessing as mp
import os
import warnings
from collections import Counter, deque
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from multiprocessing.pool import Pool
from os import cpu_count
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, Type

import chardet
import pypff
import tiktoken
from cyclopts import App, Parameter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.page_chunker import PageChunker
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.readers.file import (
    MarkdownReader,
    PandasExcelReader,
)
from llama_index.readers.file.docs import (
    HWPReader,
)
from llama_index.readers.file.epub import EpubReader
from llama_index.readers.file.ipynb import IPYNBReader
from llama_index.readers.file.mbox import MboxReader
from llama_index.readers.web import (
    SimpleWebPageReader as HTMLReader,
)
from markitdown import MarkItDown

# Suppress warnings from libraries
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PDF_PROCESSING_TIMEOUT = 300.0
_WORKER_POOL = None
_CACHED_DOCLING_CONVERTER = None


def _get_worker_pool() -> Pool:
    """Creates or retrieves the persistent pool."""
    global _WORKER_POOL
    if _WORKER_POOL is None:
        ctx = mp.get_context("spawn")
        _WORKER_POOL = ctx.Pool(processes=1, maxtasksperchild=100)
    return _WORKER_POOL


def _reset_worker_pool() -> None:
    """Zerstört den Pool hart (bei Hängern/Timeouts)."""
    global _WORKER_POOL
    if _WORKER_POOL:
        try:
            _WORKER_POOL.terminate()
            _WORKER_POOL.join()
        except Exception as e:
            logging.error(f"Error terminating pool: {e}")
        finally:
            _WORKER_POOL = None


# --- Configuration ---
# File extensions to be analyzed, case-insensitive
TARGET_EXTENSIONS = [
    ".csv",
    ".docx",
    ".epub",
    ".html",
    ".hwp",
    ".ipynb",
    ".json",
    ".mbox",
    ".md",
    ".pdf",
    ".ppt",
    ".pptm",
    ".pptx",
    ".pst",
    ".txt",
]
# CSV report filename
CSV_REPORT_FILENAME = "file_analysis_report.csv"
# Batch size for processing files (reduce if memory issues persist)
BATCH_SIZE = 50


# --- Custom Readers based on user's code ---


class PSTArchive:
    """Provides methods for manipulating a PST archive."""

    def __init__(self, file: Path | str | None = None) -> None:
        self.filepath: str | None = None
        if pypff is None:
            raise ImportError(
                "pypff is required for PST file support. Install with: pip install pypff"
            )
        self._data = pypff.file()

        if file:
            self.load(file)

    def __enter__(self) -> "PSTArchive":
        return self

    def __exit__(self, *_: Any) -> None:
        self._data.close()

    def load(self, file: Path | str) -> None:
        """Opens a PST file using libpff."""
        if isinstance(file, Path):
            file = str(file)
        self._data.open(file, "rb")
        self.filepath = file

    def folders(self, bfs: bool = True) -> Generator[pypff.folder, None, None]:
        """Generator function to iterate over the archive's folders."""
        folders = deque([self._data.root_folder])

        while folders:
            folder = folders.pop()
            yield folder
            if bfs:
                folders.extendleft(folder.sub_folders)
            else:
                folders.extend(folder.sub_folders)

    def messages(self, bfs: bool = True) -> Generator[pypff.message, None, None]:
        """Generator function to iterate over the archive's messages."""
        for folder in self.folders(bfs):
            try:
                yield from folder.sub_messages
            except OSError as exc:
                logging.debug(exc, exc_info=True)


class MarkitdownReader(BaseReader):
    def load_data(self, file: Path) -> List[Document]:
        md = MarkItDown(enable_plugins=False)
        conversion_result = md.convert(file, extract_pages=True)
        if conversion_result.pages is None:
            conversion_result.pages = [
                type(
                    "PageInfo",
                    (),
                    {"content": conversion_result.markdown, "page_number": 1},
                )()
            ]
        results = []
        for page in conversion_result.pages:
            markdown_reader = MarkdownReader()
            content = markdown_reader.remove_hyperlinks(page.content)
            content = markdown_reader.remove_images(content)
            results.append(
                Document(text=content, extra_info={"page_label": str(page.page_number)})
            )
        return results


_PROCESS_CACHED = {}


def _init_worker_converter():
    """Initialize docling converter once per worker process."""
    global _PROCESS_CACHED
    if "converter" not in _PROCESS_CACHED:

        class MDTableSerializerProvider(ChunkingSerializerProvider):
            def get_serializer(self, doc):
                return ChunkingDocSerializer(
                    doc=doc, table_serializer=MarkdownTableSerializer()
                )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(
                        document_timeout=PDF_PROCESSING_TIMEOUT
                    )
                ),
            }
        )
        _PROCESS_CACHED = {
            "converter": converter,
            "md_table_provider": MDTableSerializerProvider,
            "docling_reader": DoclingReader,
            "docling_node_parser": DoclingNodeParser,
            "page_chunker": PageChunker,
        }
    return _PROCESS_CACHED


class DoclingCustomReader(BaseReader):
    def load_data(self, file: Path) -> List[Document]:
        docs: List[Document] = []
        pdf_size_limit = 500

        if (
            "pdf" in file.suffix.lower()
            and file.stat().st_size / 1024**2 > pdf_size_limit
        ):
            logging.warning(
                f"{file.name} larger than {pdf_size_limit} MB. File was {round(file.stat().st_size / 1024**2, 2)} MB."
            )
            return docs

        cached = _init_worker_converter()

        try:
            reader = cached["docling_reader"](
                export_type=DoclingReader.ExportType.JSON,
                doc_converter=cached["converter"],
            )
            json_docs = reader.load_data(file)

            node_parser = cached["docling_node_parser"](
                chunker=cached["page_chunker"](
                    serializer_provider=cached["md_table_provider"](),
                )
            )

            nodes = node_parser.get_nodes_from_documents(documents=json_docs)

            docs = [
                Document(
                    doc_id=node.id_,
                    text=node.get_content(),
                    metadata={"page_label": str(idx + 1)},
                )
                for idx, node in enumerate(nodes)
            ]

            if not docs:
                docs = [
                    Document(doc_id=doc.id_, text="") for _, doc in enumerate(json_docs)
                ]

        except Exception as e:
            logging.exception("Failed to transform into docling", exc_info=e)

        return docs


class PSTReader(BaseReader):
    md = MarkItDown()

    def extract_text(self, message) -> str:
        """Extract text content from a PST email message."""
        text = ""
        if message.plain_text_body:
            text = message.plain_text_body
        elif hasattr(message, "html_body") and message.html_body:
            text = self.md.convert(BytesIO(message.html_body)).markdown
        elif hasattr(message, "rtf_body") and message.rtf_body:
            text = message.rtf_body

        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")

        if text:
            sent_date = (
                message.client_submit_time.strftime("%d.%m.%Y %H:%M Uhr") + "\n"
                if message.client_submit_time
                else ""
            )
            sender_name = message.sender_name + "\n" if message.sender_name else ""
            subject = message.subject + "\n" if message.subject else ""
            header = sent_date + sender_name + subject + "\n"
            text = text.replace("\\_", "")
            text = header + text

        return text

    def load_data(self, file: Path) -> List[Document]:
        if pypff is None:
            logging.error("pypff not installed. Cannot process PST files.")
            return []

        pst_archive = PSTArchive(file)
        mails = []
        for message in pst_archive.messages():
            try:
                text = self.extract_text(message)
                if text:
                    mails.append(Document(text=text))
            except OSError:
                pass
            except Exception as e:
                logging.exception("Failed to transform pst file", exc_info=e)

        logging.info(f"Extracted {len(mails)} mails.")
        return mails


class MSReader(BaseReader):
    """Reader for Microsoft Word (.docx) and PowerPoint (.pptx, .ppt, .pptm) files."""

    def load_data(self, file: Path, **kwargs: Any) -> List[Document]:
        md = MarkItDown(enable_plugins=False)
        try:
            content = md.convert(file).markdown
            # Simple text conversion, could be elaborated with MarkdownReader logic if needed
            return [Document(text=content, metadata={"file_name": file.name})]
        except Exception as e:
            logging.error(f"Failed to process MS file {file.name}: {e}")
            return []


class ExcelReader(BaseReader):
    """Reader for Excel files (.xlsx, .xls)."""

    def load_data(self, file: Path, **kwargs: Any) -> List[Document]:
        try:
            return PandasExcelReader(concat_rows=True).load_data(file, **kwargs)
        except Exception as e:
            logging.error(f"Failed to process Excel file {file.name}: {e}")
            return []


# --- File Reader Mapping ---
# Maps file extensions to their corresponding reader classes.
FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".hwp": HWPReader,
    ".pdf": DoclingCustomReader,
    ".docx": DoclingCustomReader,
    ".pptx": DoclingCustomReader,
    ".ppt": DoclingCustomReader,
    ".pptm": DoclingCustomReader,
    ".jpg": DoclingCustomReader,
    ".png": DoclingCustomReader,
    ".jpeg": DoclingCustomReader,
    ".csv": DoclingCustomReader,
    ".epub": EpubReader,
    ".md": DoclingCustomReader,
    ".mbox": MboxReader,
    ".ipynb": IPYNBReader,
    ".xlsx": DoclingCustomReader,
    ".xls": DoclingCustomReader,
    ".xltx": DoclingCustomReader,
    ".pst": PSTReader,
}

# --- Core Functions ---


def get_tokenizer():
    """Initializes and returns the tiktoken tokenizer."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        logging.info("Default tokenizer not found. Downloading 'cl100k_base'.")
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, tokenizer) -> int:
    """Counts tokens in a string."""
    if not text or not tokenizer:
        return 0
    return len(tokenizer.encode(text, disallowed_special=()))


def load_file_to_text(file_path: Path) -> str:
    """
    Loads a file using the appropriate LlamaIndex reader and returns its text content.
    """
    extension = file_path.suffix.lower()
    reader_cls = FILE_READER_CLS.get(extension)
    documents: List[Document] = []

    try:
        if reader_cls:
            if reader_cls == HTMLReader:
                documents = HTMLReader().load_data([file_path.as_uri()])
            else:
                reader_instance = reader_cls()
                if extension == ".csv":
                    encoding = chardet.detect(file_path.read_bytes())["encoding"]
                    documents = reader_instance.load_data(
                        file_path, pandas_config={"encoding": encoding}
                    )
                else:
                    documents = reader_instance.load_data(file_path)
        else:
            encoding = chardet.detect(file_path.read_bytes())["encoding"] or "utf-8"
            content = file_path.read_text(encoding=encoding, errors="ignore")
            documents = [Document(text=content)]

    except Exception as e:
        logging.error(f"Could not read or process file {file_path.name}: {e}")
        return ""

    return "\n\n".join(doc.get_content() for doc in documents if doc.get_content())


def process_file(file_path_str: str) -> Dict[str, Any]:
    """Worker function to process a single file. Designed to be run in a separate process."""
    # This function is designed to be self-contained for multiprocessing
    logging.info(f"Processing: {os.path.basename(file_path_str)}")
    file_path = Path(file_path_str)
    ext = file_path.suffix.lower()

    # Initialize tokenizer within the worker process
    tokenizer = get_tokenizer()

    try:
        size_bytes = file_path.stat().st_size
    except FileNotFoundError:
        logging.warning(f"File not found during size check: {file_path}. Skipping.")
        return None

    text_content = load_file_to_text(file_path)
    tokens = count_tokens(text_content, tokenizer) if text_content else 0

    # Clear text_content immediately to free memory
    del text_content

    return {
        "path": file_path_str,
        "ext": ext,
        "size_bytes": size_bytes,
        "tokens": tokens,
    }


def batch_iterator(items: List[str], batch_size: int) -> Iterator[List[str]]:
    """Yield successive batches from items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def append_to_csv(
    data: List[Dict[str, Any]], filename: str, write_header: bool = False
):
    """Appends data to a CSV file incrementally."""
    if not data:
        return

    mode = "w" if write_header else "a"
    try:
        with open(filename, mode, newline="", encoding="utf-8") as csvfile:
            fieldnames = ["file_path", "size_mb", "token_count"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(data)
    except IOError as e:
        logging.error(f"Failed to write to CSV file {filename}: {e}")


def save_per_file_report_to_csv(report_data: List[Dict[str, Any]], filename: str):
    """Saves the per-file analysis results to a CSV file."""
    if not report_data:
        logging.warning("No per-file data to save to CSV.")
        return

    logging.info(f"Saving per-file report to {filename}...")
    try:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["file_path", "size_mb", "token_count"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_data)
        logging.info(f"Successfully saved per-file report to {filename}.")
    except IOError as e:
        logging.error(f"Failed to write to CSV file {filename}: {e}")


def save_summary_report_to_csv(summary_data: List[Dict[str, Any]], filename: str):
    """Saves the summary analysis results to a CSV file."""
    if not summary_data:
        logging.warning("No summary data to save to CSV.")
        return

    logging.info(f"Saving summary report to {filename}...")
    try:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["file_type", "file_count", "total_tokens"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        logging.info(f"Successfully saved summary report to {filename}.")
    except IOError as e:
        logging.error(f"Failed to write to CSV file {filename}: {e}")


app = App()


def run_analysis(root_dir: Path, batch_size: int = BATCH_SIZE):
    """
    Main function to find files, analyze them in parallel batches, and generate reports.
    Processes files in batches to reduce memory usage.
    """
    logging.info(f"Starting file analysis in: {root_dir}")

    all_files = []
    for ext in TARGET_EXTENSIONS:
        all_files.extend(glob.glob(f"{root_dir}/**/*{ext}", recursive=True))
        all_files.extend(glob.glob(f"{root_dir}/**/*{ext.upper()}", recursive=True))

    all_files = sorted(set(all_files))

    if not all_files:
        logging.warning("No files found with the specified extensions. Exiting.")
        return

    max_workers = max(1, int(cpu_count() * 0.75))
    logging.info(
        "Found %d files to analyze. Using up to %d processes (75%% of available cores).",
        len(all_files),
        max_workers,
    )
    logging.info(
        f"Processing in batches of {batch_size} files to optimize memory usage."
    )

    # --- Initialize CSV with header ---
    per_file_csv = "file_analysis_report.csv"
    append_to_csv([], per_file_csv, write_header=True)

    # --- Aggregation counters ---
    extension_counts = Counter()
    token_counts_by_ext = Counter()
    total_tokens = 0

    # Track top files for reporting (limit memory usage)
    top_files_by_size = []
    top_files_by_tokens = []
    max_top_files = 20

    # --- Process files in batches ---
    processed_count = 0
    for batch_num, batch in enumerate(batch_iterator(all_files, batch_size), 1):
        logging.info(
            f"Processing batch {batch_num}/{(len(all_files) + batch_size - 1) // batch_size}"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(process_file, batch))

        batch_results = [res for res in batch_results if res is not None]

        # Process batch results
        batch_report_data = []
        for res in batch_results:
            # Update counters
            extension_counts[res["ext"]] += 1
            token_counts_by_ext[res["ext"]] += res["tokens"]
            total_tokens += res["tokens"]

            # Prepare per-file data
            file_data = {
                "file_path": res["path"],
                "size_mb": round(res["size_bytes"] / 1024 / 1024, 4),
                "token_count": res["tokens"],
            }
            batch_report_data.append(file_data)

            # Track top files (memory-efficient approach)
            if len(top_files_by_size) < max_top_files:
                top_files_by_size.append(file_data)
                top_files_by_size.sort(key=lambda x: x["size_mb"], reverse=True)
            elif file_data["size_mb"] > top_files_by_size[-1]["size_mb"]:
                top_files_by_size[-1] = file_data
                top_files_by_size.sort(key=lambda x: x["size_mb"], reverse=True)

            if len(top_files_by_tokens) < max_top_files:
                top_files_by_tokens.append(file_data)
                top_files_by_tokens.sort(key=lambda x: x["token_count"], reverse=True)
            elif file_data["token_count"] > top_files_by_tokens[-1]["token_count"]:
                top_files_by_tokens[-1] = file_data
                top_files_by_tokens.sort(key=lambda x: x["token_count"], reverse=True)

        # Append batch results to CSV immediately
        append_to_csv(batch_report_data, per_file_csv, write_header=False)

        processed_count += len(batch_results)
        logging.info(f"Processed {processed_count}/{len(all_files)} files")

        # Clear batch data to free memory
        del batch_results
        del batch_report_data

    # --- Generate summary report ---
    summary_report_data = [
        {
            "file_type": ext,
            "file_count": count,
            "total_tokens": token_counts_by_ext[ext],
        }
        for ext, count in extension_counts.items()
    ]

    save_summary_report_to_csv(summary_report_data, "file_type_summary.csv")

    # --- Console Reporting ---
    print("\n" + "=" * 50)
    print("          FILE ANALYSIS REPORT")
    print("=" * 50 + "\n")

    # 1. File Extension Distribution
    print("--- File Extension Distribution ---")
    for ext, count in sorted(extension_counts.items()):
        print(f"{ext:<10}: {count} file(s)")
    print("\n" + "-" * 50 + "\n")

    # 2. Tokens by File Type
    print("--- Tokens by File Type ---")
    sorted_tokens_by_ext = sorted(
        summary_report_data, key=lambda x: x["total_tokens"], reverse=True
    )
    for item in sorted_tokens_by_ext:
        print(f"{item['file_type']:<10}: {item['total_tokens']:,} tokens")
    print("\n" + "-" * 50 + "\n")

    # 3. Files by Size (showing top 20)
    print(f"--- Top {max_top_files} Files by Size (Descending) ---")
    for item in top_files_by_size:
        print(f"{item['size_mb']:.2f} MB - {item['file_path']}")
    print("\n" + "-" * 50 + "\n")

    # 4. Token Counts
    print(f"--- Top {max_top_files} Files by Token Count (Descending) ---")
    for item in top_files_by_tokens:
        print(f"{item['token_count']:,} tokens - {item['file_path']}")

    print("\n" + "-" * 50 + "\n")
    print("--- Token Count Summary ---")
    print(f"Total estimated tokens across all files: {total_tokens:,}")

    print("\n" + "-" * 50 + "\n")

    # 5. Costs
    print("--- Estimated costs (2025-09-18) ---")
    print(
        f"Total estimated costs for initial ingestion: €{round(total_tokens / 1000 * 0.000104, 4)}"
    )

    print("\n" + "-" * 50 + "\n")
    # --- CSV Report Confirmation ---
    print("--- CSV Reports Saved ---")
    print("Detailed per-file report saved to: file_analysis_report.csv")
    print("Summary report saved to: file_type_summary.csv")

    print("\n" + "=" * 50)
    print("            END OF REPORT")
    print("=" * 50 + "\n")


@app.default
def main(
    directory: Annotated[
        Path,
        Parameter(help="The directory to analyze. Defaults to the current directory."),
    ],
    excel: Annotated[
        bool,
        Parameter(help="If excel files should be included. Defaults to False"),
    ] = False,
    batch_size: Annotated[
        int,
        Parameter(
            help="Number of files to process per batch. Lower values use less memory."
        ),
    ] = BATCH_SIZE,
):
    """Analyzes a directory for file distribution, size, and token counts."""
    if excel:
        logging.info("Including excel files.")
        TARGET_EXTENSIONS.extend(
            [
                ".xls",
                ".xlsx",
            ]
        )
    run_analysis(directory.resolve(), batch_size=batch_size)


if __name__ == "__main__":
    app()
