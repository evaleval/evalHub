from __future__ import annotations
from argparse import ArgumentParser
import json
from enum import Enum
from pathlib import Path
from typing import List, Union

from eval_converters.helm.adapter import HELMAdapter
from schema.eval_types import (
    EvaluatorRelationship,
    EvaluationLog,
    SourceMetadata
)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--log_dirpath', type=str, default='tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2', help="Path to directory with single evaluaion or multiple evaluations to convert")
    parser.add_argument('--huggingface_dataset', type=str)
    parser.add_argument('--output_dir', type=str, default='tests/data/helm')
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

class HELMEvalLogConverter:
    def __init__(self, log_dirpath: str | Path, output_dir: str = 'unified_schema/helm'):
        '''
        HELM generates log file for an evaluation.
        '''
        self.log_dirpath = Path(log_dirpath)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_unified_schema(self, source_metadata: SourceMetadata = None) -> Union[EvaluationLog, List[EvaluationLog]]:
        return HELMAdapter().transform_from_directory(self.log_dirpath, source_metadata=source_metadata)

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

    helm_converter = HELMEvalLogConverter(
        log_dirpath=args.log_dirpath,
        output_dir=args.output_dir
    )
    
    source_metadata = SourceMetadata(
        source_organization_name=args.source_organization_name,
        source_organization_url=args.source_organization_url,
        source_organization_logo_url=args.source_organization_logo_url,
        evaluator_relationship=EvaluatorRelationship(args.evaluator_relationship)
    )

    unified_output = helm_converter.convert_to_unified_schema(source_metadata)

    if unified_output and isinstance(unified_output, EvaluationLog):
        output_filename = f'{str(unified_output.evaluation_id).replace('/', '_')}.json'
        helm_converter.save_to_file(unified_output, output_filename)
    elif unified_output and isinstance(unified_output, List):
        for single_unified_output in unified_output:
            output_filename = f'{str(single_unified_output.evaluation_id).replace('/', '_')}.json'
            helm_converter.save_to_file(single_unified_output, output_filename)
    else:
        print("Missing unified schema result!")