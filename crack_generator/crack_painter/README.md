# Crack Painter

Interactively draw lines on TIFF images (or other formats) to generate realistic cracks.
This tool leverages concepts and visual styles from the `sutadasuto/syncrack_generator` repository.

## Installation

1.  Clone this repository.
2.  It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Install the package itself (for the `crack-painter` command):
    ```bash
    pip install .
    ```
5.  **Important**: Ensure the `sutadasuto-syncrack_generator` code is accessible. If it's not installed as a package, you might need to:
    *   Place its directory (e.g., `sutadasuto-syncrack_generator/`) alongside the `crack_painter/` directory. The `main.py` has commented-out lines to help locate it.
    *   Or, add `sutadasuto-syncrack_generator/` to your `PYTHONPATH`.

## Usage

```bash
crack-painter path/to/your_image.tif -o path/to/output_cracked_image.png