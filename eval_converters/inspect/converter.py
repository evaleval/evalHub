from __future__ import annotations
from argparse import ArgumentParser
import json
from enum import Enum
from pydantic_core.core_schema import model_schema
from pathlib import Path

from eval_converters.inspect.adapter import InspectAIAdapter
from schema.eval_types import EvaluationLog
from schema.eval_types import (
    EvaluatorRelationship,
    EvaluationLog,
    SourceMetadata
)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--log_path', type=str, default='tests/data/inspect/data.json')
    parser.add_argument('--huggingface_dataset', type=str)
    parser.add_argument('--output_dir', type=str, default='tests/data/inspect')
    parser.add_argument('--source_organization_name', type=str, default='Unknown', help='Orgnization which pushed evaluation to the evalHub.')
    parser.add_argument('--evaluator_relationship', type=str, default='other', help='Relationship of evaluation author to the model', choices=['first_party', 'third_party', 'collaborative', 'other'])
    parser.add_argument('--source_organization_url', type=str, default=None)
    parser.add_argument('--source_organization_logo_url', type=str, default=None)


    args = parser.parse_args()
    return args


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

class InspectEvalLogConverter:
    def __init__(self, log_path: str | Path, output_dir: str = 'unified_schema/inspect_ai'):
        '''
        InspectAI generates log file for an evaluation.
        '''
        self.log_path = Path(log_path)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_unified_schema(self, source_metadata: SourceMetadata) -> EvaluationLog:
        return InspectAIAdapter().transform_from_file(self.log_path, source_metadata=source_metadata)

    def save_to_file(self, unified_eval_log: EvaluationLog, output_filename: str) -> bool:
        try:
            json_str = unified_eval_log.model_dump_json(indent=2)

            with open(f'{self.output_dir}/{output_filename}', 'w') as json_file:
                json_file.write(json_str)

            print(f'Unified eval log was successfully saved to {output_filename} file.')
        except Exception as e:
            print(f"Problem with saving unified eval log to file: {e}")
            raise e

    def save_to_hf_datasets(self, unified_eval_log: EvaluationLog) -> bool:
        # TODO
        pass


if __name__ == '__main__':
    args = parse_args()

    inspect_converter = InspectEvalLogConverter(
        log_path=args.log_path,
        output_dir=args.output_dir
    )
    
    source_metadata = SourceMetadata(
        source_organization_name=args.source_organization_name,
        source_organization_url=args.source_organization_url,
        source_organization_logo_url=args.source_organization_logo_url,
        evaluator_relationship=EvaluatorRelationship(args.evaluator_relationship)
    )

    unified_output: EvaluationLog = inspect_converter.convert_to_unified_schema(source_metadata)
    if unified_output:
        output_filename = f'{str(unified_output.evaluation_id).replace('/', '_')}.json'
        inspect_converter.save_to_file(unified_output, output_filename)
    else:
        print("Missing unified schema result!")