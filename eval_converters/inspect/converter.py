from __future__ import annotations

import json
from pydantic_core.core_schema import model_schema
from pathlib import Path
from typing import List

from eval_converters.inspect.adapter import InspectAIAdapter
from schema.eval_types import EvaluationResult

class InspectEvalLogConverter:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)

        self.output_dir = Path("inspectai/unified_schema", log_path.split('.')[0])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_unified_schema(self) -> List[EvaluationResult]:
        return InspectAIAdapter().transform_from_file(self.log_path)
    
    def save(self, unified_eval_log: List[EvaluationResult]):
        json_data = json.dumps(
            [item.model_dump() for item in unified_eval_log], 
            indent=2
        )
        with open(f'{self.output_dir}/unified_log.json', 'w') as json_file:
            json.dump(json_data, json_file)
    
if __name__ == '__main__':
    inspect_log_filepath = 'eval_converters/inspect/data.json'
    inspect_converter = InspectEvalLogConverter(log_path=inspect_log_filepath)
    unified_output: EvaluationResult = inspect_converter.convert_to_unified_schema()
    inspect_converter.save(unified_output)