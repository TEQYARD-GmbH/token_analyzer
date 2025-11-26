# /// script
# dependencies = [
#     "llama-index",
#     "pandas",
#     "openpyxl",
#     "python-docx",
#     "markitdown[docx,pptx,pdf]",
#     "tiktoken",
#     "chardet",
#     "llama-index-readers-file",
#     "llama-index-readers-web",
#     "beautifulsoup4",
#     "PyMuPDF",
#     "llama-index-readers-json>=0.4.1",
#     "pikepdf",
#     "cyclopts",
#     "xlrd"
# ]
# ///

import csv
import glob
import logging
import os
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, Type

import chardet
import tiktoken
from cyclopts import App, Parameter

# Suppress warnings from libraries
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- LlamaIndex and other library imports ---
# These are heavy imports, so we do them carefully.
try:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document
    from llama_index.readers.file import (
        MarkdownReader,
        PandasCSVReader,
        PandasExcelReader,
    )
    from llama_index.readers.file.docs import (
        HWPReader,
    )
    from llama_index.readers.file.epub import EpubReader
    from llama_index.readers.file.ipynb import IPYNBReader
    from llama_index.readers.file.mbox import MboxReader
    from llama_index.readers.json import JSONReader
    from llama_index.readers.web import (
        SimpleWebPageReader as HTMLReader,
    )  # Using SimpleWebPageReader for HTML
    from markitdown import MarkItDown
except ImportError as e:
    logging.error(
        f"Failed to import a required library. Please add package to list of dependencies. Error: {e}"
    )
    exit(1)

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
    ".txt",
]
# CSV report filename
CSV_REPORT_FILENAME = "file_analysis_report.csv"
# Batch size for processing files (reduce if memory issues persist)
BATCH_SIZE = 50


# --- Custom Readers based on user's code ---


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
    ".pdf": MSReader,
    ".docx": MSReader,
    ".pptx": MSReader,
    ".ppt": MSReader,
    ".pptm": MSReader,
    ".csv": PandasCSVReader,
    ".epub": EpubReader,
    ".md": MarkdownReader,
    ".mbox": MboxReader,
    ".ipynb": IPYNBReader,
    ".xlsx": ExcelReader,
    ".xls": ExcelReader,
    ".json": JSONReader,
    ".html": HTMLReader,
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
