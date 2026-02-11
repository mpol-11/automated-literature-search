# automated-literature-search
Automated pipeline for searching scientific papers, downloading open-access PDFs, and generating transformer-based summaries.

# Automated Literature Search and Summarization

# ⚠️ Status: Work in Progress
This project is actively being developed. Some features may be incomplete, unstable, or subject to change. Bug fixes and improvements are ongoing.

## Overview

This project implements an automated pipeline for searching scientific literature, downloading open-access research papers, and generating concise summaries using transformer-based language models. It is designed for local execution from the command line and can be used for:
1) Exploratory research
2) Literature reviews
3) Rapid paper screening


The architecture is modular, making it easy to extend, modify, or integrate into larger research systems.

## Features

- Search scientific literature by topic, author, or keywords

- Automatically download open-access PDFs (arXiv, bioRxiv, medRxiv, etc.)

- Provide direct URLs when PDFs cannot be downloaded

- Summarize papers using a biomedical PEGASUS transformer model

- Modular design separating search, download, and summarization logic

- Runs locally without cloud or notebook dependencies


## Requirements

- Python 3.9+
- Internet connection (for paper retrieval + model download)

Note: The summarization model currently used is large and may take several minutes to download the first time it is used.

## Installation

Clone the repository and install dependencies:

git clone https://github.com/mpol-11/automated-literature-search.git

cd automated-literature-search

python3 -m pip install -r requirements.txt

## Usage

Run from the project root directory:

python3 src/search_app.py


You will be prompted to:

1) Enter a topic, author, or keyword
2) Choose whether to summarize abstracts
3) Select papers from results
4) Download PDFs or receive direct links
5) Generate summaries of selected papers

## How the Pipeline Works

1. Literature Search: 
Queries scientific sources using user-provided keywords or filters.

2. Paper Retrieval: 
Detects open-access versions and downloads PDFs locally when available. Otherwise returns direct links.

3. Summarization: 
Extracts text from PDFs, detects major sections, and generates concise summaries using a transformer model.

## Limitations

- Only open-access papers can be downloaded automatically
- PDF text extraction accuracy depends on formatting quality
- Generated summaries are meant for quick understanding, not full comprehension
- Transformer models may require significant RAM and compute
- Some edge cases may produce incomplete or incorrect results (actively being improved)

## Planned Improvements

- Command-line arguments instead of interactive prompts
- Additional filtering options
- Support for more literature databases
- Improved summarization quality
- Export summaries to Markdown or JSON
- Unit tests + CI integration
- Integration with scientific writing tools

## Contributing

This project is currently under active development. Suggestions, issues, and improvements are welcome. Expect frequent updates and structural changes.
