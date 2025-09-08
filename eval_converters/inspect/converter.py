from __future__ import annotations
from argparse import ArgumentParser
import json
from pydantic_core.core_schema import model_schema
from pathlib import Path
from typing import List

from eval_converters.inspect.adapter import InspectAIAdapter
from schema.eval_types import EvaluationResult

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--log_path', type=str, default='tests/data/inspectai/data.json')
    parser.add_argument('--output_dir', type=str, default='unified_schema/inspect_ai')

    args = parser.parse_args()
    return args

class InspectEvalLogConverter:
    def __init__(self, log_path: str | Path, output_dir: str = 'unified_schema/inspect_ai'):
        '''
        InspectAI generates log file for an evaluation.
        '''
        self.log_path = Path(log_path)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_unified_schema(self) -> List[EvaluationResult]:
        return InspectAIAdapter().transform_from_file(self.log_path)
    
    def save(self, unified_eval_log: List[EvaluationResult], output_filename: str):
        json_data = json.dumps(
            [item.model_dump() for item in unified_eval_log], 
            indent=2
        )
        with open(f'{self.output_dir}/{output_filename}', 'w') as json_file:
            json.dump(json_data, json_file)


if __name__ == '__main__':
    args = parse_args()

    inspect_converter = InspectEvalLogConverter(
        log_path=args.log_path,
        output_dir=args.output_dir
    )
    
    unified_output: EvaluationResult = inspect_converter.convert_to_unified_schema()
    output_filename = f'{str(unified_output.evaluation_id)}.json'
    inspect_converter.save(unified_output, output_filename)