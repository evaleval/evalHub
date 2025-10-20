import os

from typing import List
from inspect_ai.log import EvalLog, EvalSpec, EvalStats, read_eval_log

from dacite import from_dict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from schema import SCHEMA_VERSION
from schema.eval_types import (
    DetailedEvaluationResult,
    EvaluationLog,
    EvaluationResult,
    EvaluationSource,
    EvaluationSourceType,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    SourceData,
    SourceMetadata
)

from eval_converters.common.adapter import AdapterMetadata, BaseEvaluationAdapter, SupportedLibrary
from eval_converters.common.error import AdapterError

class InspectAIAdapter(BaseEvaluationAdapter):
    """
    Adapter for transforming evaluation outputs from the Inspect AI library into the unified schema format.
    """

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
			name="InspectAdapter",
			version="0.0.1",
			description="Adapter for transforming HELM evaluation outputs to unified schema format"
		)

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.INSPECT_AI
        
    def transform_from_directory(self, dir_path: Union[str, Path]):
        raise NotImplementedError("Inspect AI adapter do not support loading logs from directory!")

    def transform_from_file(self, file_path: Union[str, Path], source_metadata: SourceMetadata = None) -> Union[EvaluationLog, List[EvaluationLog]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File path {file_path} does not exists!')
        
        try:
            file_path = Path(file_path)
            eval_log: EvalLog = self._load_file(file_path)
            return self.transform(eval_log, source_metadata)
        except AdapterError as e:
            raise e
        except Exception as e:
            raise AdapterError(f"Failed to load file {file_path}: {str(e)} for InspectAIAdapter")
    
    def _get_score(self, response: str, ground_truth: str, metric_name: str) -> float:
        if metric_name.strip().lower() == 'accuracy':
            score = float(response.strip().lower() == ground_truth.strip().lower())
            
            if not score:
                score = float(response.endswith(ground_truth))
            
            return score
        else:
            raise NotImplemented('Metric other than accuracy do not supported at this moment.')

    def _transform_single(self, raw_data: EvalLog, source_metadata: SourceMetadata) -> EvaluationLog:
        eval_spec: EvalSpec = raw_data.eval
        eval_stats: EvalStats = raw_data.stats

        retrieved_timestamp = eval_stats.started_at or eval_spec.created
        
        source_data = SourceData(
            dataset_name=eval_spec.dataset.name.split('/')[-1],
            hf_repo=eval_spec.dataset.location,
            samples_number=eval_spec.dataset.samples,
            sample_ids=eval_spec.dataset.sample_ids
        )

        evaluation_source = EvaluationSource(
            evaluation_source_name='inspect_ai',
            evaluation_source_type=EvaluationSourceType.evaluation_platform
        )

        model_path = eval_spec.model
        if raw_data.samples:
            model_name = raw_data.samples[0].output.model
            model_path_parts = model_path.split('/')

            if model_path_parts[-1] in model_name:
                model_path_parts[-1] = model_name

            model_path = '/'.join(model_path_parts)

        self._check_if_model_is_on_huggingface(model_path)

        model_info = ModelInfo(
            name=model_path,
            developer=eval_spec.model.split('/')[0],
            inference_platform="/".join(eval_spec.model.split('/')[:-1])
        )

        results = raw_data.results
        evaluation_results = []

        generation_config = {
            gen_config: value 
            for gen_config, value in vars(eval_spec.model_generate_config).items() if value is not None
        }

        for scorer_results in results.scores:
            scorer_name = scorer_results.scorer
            for metric in scorer_results.metrics:
                metric_info = scorer_results.metrics[metric]
                if metric_info.name != 'stderr':
                    evaluation_results.append(EvaluationResult(
                        evaluation_name=scorer_name,
                        evaluation_timestamp=eval_stats.completed_at,
                        metric_config=MetricConfig(
                            evaluation_description=metric_info.name,
                            lower_is_better=False # probably there is no access to such info
                        ),
                        score_details=ScoreDetails(
                            score=metric_info.value
                        ),
                        generation_config=generation_config
                    ))

        detailed_evaluation_results = []
        for sample in raw_data.samples:
            if sample.scores:
                response = sample.scores.get('choice').answer
            else:
                response = sample.output.choices[0].message.content
            
            detailed_evaluation_results.append(
                DetailedEvaluationResult(
                    sample_id=sample.id,
                    input=sample.input,
                    ground_truth=sample.target,
                    response=response,
                    choices=sample.choices
                )
            )

        evaluation_id = f'inspect_ai/{model_path}/{eval_spec.dataset.name}/{retrieved_timestamp}'

        return EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_data=source_data,
            evaluation_source=evaluation_source,
            source_metadata=source_metadata,
            model_info=model_info,
            evaluation_results=evaluation_results,
            detailed_evaluation_results=detailed_evaluation_results
        )
        
    def _load_file(self, file_path) -> EvalLog:
        return read_eval_log(file_path)