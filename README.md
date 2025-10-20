# EvalHub Infrastructure

This repository provides a unified and extensible framework for running and organizing evaluations across multiple LLM evaluation tools such as [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness), [`HELM`](https://github.com/stanford-crfm/helm), etc.

## Prerequisites

- Python 3.12 or higher
- [`uv`](https://docs.astral.sh/uv/)

## Installation

- Install the required dependencies:

```bash
uv sync
```

## Scripts

### Inspect
Convert eval log from Inspect AI into json format with following command:

```bash
uv run inspect log convert path_to_eval_file_generated_by_inspect --to json --output-dir inspect_json
```

Then we can convert Inspect evaluation log into unified schema via eval_converters/inspect/converter.py. Conversion for example data can be generated via below script: 

```bash
uv run python3 -m eval_converters.inspect.converter
```

Documentation for conversion of your own Inspect evaluation log into unified is available below:

```bash
usage: converter.py [-h] [--log_path LOG_PATH]
                    [--huggingface_dataset HUGGINGFACE_DATASET]
                    [--output_dir OUTPUT_DIR]
                    [--source_organization_name SOURCE_ORGANIZATION_NAME]
                    [--evaluator_relationship {first_party,third_party,collaborative,other}]
                    [--source_organization_url SOURCE_ORGANIZATION_URL]
                    [--source_organization_logo_url SOURCE_ORGANIZATION_LOGO_URL]

options:
  -h, --help            show this help message and exit
  --log_path LOG_PATH
  --huggingface_dataset HUGGINGFACE_DATASET
  --output_dir OUTPUT_DIR
  --source_organization_name SOURCE_ORGANIZATION_NAME
                        Orgnization which pushed evaluation to the evalHub.
  --evaluator_relationship {first_party,third_party,collaborative,other}
                        Relationship of evaluation author to the model
  --source_organization_url SOURCE_ORGANIZATION_URL
  --source_organization_logo_url SOURCE_ORGANIZATION_LOGO_URL
```