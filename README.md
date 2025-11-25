# Token Analyzer

This script analyzes a directory of files, counts the tokens in each file, and generates reports on file types, sizes, and token counts.

## Features

- Analyzes a variety of file types, including documents, spreadsheets, presentations, and code.
- Uses `tiktoken` to count tokens.
- Generates two CSV reports:
    - `file_analysis_report.csv`: A detailed report on each file.
    - `file_type_summary.csv`: A summary report on file types.
- Prints a summary report to the console.

## Usage

1. Place the script in the root directory of the project you want to analyze.
2. Run the script:

- `uv run token_analyzer.py` to analyze the current folder
- `uv run token_analyzer.py /path/to/folder` to analyze a different one
- `uv run token_analyzer.py /path/to/folder True` to include Excel files
