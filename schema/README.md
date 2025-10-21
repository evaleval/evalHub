# Evaluation Format Schema

## Generate Python Types from JSON Schema

To generate or update Python types from the JSON schema, you can run the following command in the current directory:

```bash
uv add datamodel-code-generator
cd schema
uv run datamodel-codegen --input eval.schema.json --output eval_types.py --class-name EvaluationLog --output-model-type pydantic_v2.BaseModel --input-file-type jsonschema
```

## Example Data

Please refer to [this file](./eval.example.json) for a minimal data example adhering to the evaluation format schema.

## Leaderboard Format Schema

The leaderboard format schema is defined in [leaderboard.schema.json](./leaderboard.schema.json). It is used to store and validate leaderboard-style data.
The schema allows for "scores" both in a continuous format (e.g. accuracy) or level-based (e.g. low/medium/high).

For level-based data, actual scores are stored as integers that are mapped to the corresponding level names. Scores are values from 0 to N or -1 to N (if has_unknown_level is True). For presentation purposes, the values are mapped to the level names in the level_names array.