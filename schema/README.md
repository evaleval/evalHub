# Evaluation Format Schema

## Generate Python Types from JSON Schema

To generate or update Python types from the JSON schema, you can run the following command in the current directory:

```bash
datamodel-codegen --input eval.schema.json --output eval_types.py --class-name EvaluationResult --output-model-type pydantic_v2.BaseModel
```
