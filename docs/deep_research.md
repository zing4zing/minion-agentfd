# Deep Research

This document explains how to use the Deep Research feature in this project.

## Installation

To use Deep Research, you need to install the required dependencies:

### Python Dependencies

Install the deep research extras:

```bash
pip install -e '.[deep-research]'
```

### System Dependencies

Some features require Pandoc and pdfLaTeX. Please install them according to your operating system:

- **macOS**:
  1. Install Pandoc:
     ```bash
     brew install pandoc
     ```
  2. Install pdfLaTeX (via BasicTeX):
     ```bash
     brew install basictex
     ```

- **Ubuntu/Debian**:
  1. Install Pandoc:
     ```bash
     sudo apt-get install pandoc
     ```
  2. Install pdfLaTeX:
     ```bash
     sudo apt-get install texlive-xetex
     ```

- **Windows**:
  1. Download and install Pandoc from [pandoc.org](https://pandoc.org/installing.html)
  2. Download and install MiKTeX (for pdfLaTeX) from [miktex.org](https://miktex.org/download)

## Usage

1. Ensure all dependencies are installed.
2. Refer to example_deep_research.py,example_deep_research_pdf.py,example_deep_research_html.py  for how to invoke Deep Research features.

For more details, see the main README or contact the project maintainers. 