# Paradox Localisation MTL Translator

This Python script provides an easy and configurable solution for translating YAML files using various LLMs.

## Prerequisites

*   Python 3.7+
*   Required Python libraries:
    *   `PyYAML`
    *   `google-generativeai`
    *   `openai`
    *   `requests`
    *   `tqdm`
    *   `rich`

    Install them using pip:
    ```bash
    pip install pyyaml google-generativeai openai requests tqdm rich
    ```

## Usage

Run the script from your terminal:

*   **Translate a single file:**
    ```bash
    python MTL.py path/to/your/file.yml [options]
    ```
*   **Translate all matching files in a folder:**
    ```bash
    python MTL.py path/to/your/folder [options]
    ```
*   **See help for all options:**
    ```bash
    python MTL.py --help
    ```

**Common Options:**

*   `--output <path>` or `-o <path>`: Specify output file or directory.
*   `--config <path>`: Path to your `config.yaml` file.
*   `--pattern "*.yml,*.yaml"`: Comma-separated file patterns to match.
*   `--recursive` or `-r`: Process folders recursively.
*   `--no-ui`: Disable the Rich-based UI and use plain console output.
*   `--threads <N>`: Set the maximum number of worker threads for parallel file/batch processing.
