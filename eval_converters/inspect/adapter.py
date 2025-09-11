import os

from typing import List
from inspect_ai.log import EvalLog, EvalSample, EvalSpec, read_eval_log
from inspect_ai.model import GenerateConfig

from dacite import from_dict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from schema import SCHEMA_VERSION
from schema.eval_types import (
	EvaluationResult,
	ModelInfo,
    Dimensions,
	GenerationArgs,
	InferenceSettings,
    InstructionPhrasing,
	Instance,
	Output,
	Evaluation,
	EvaluationMethod,
	PromptConfig,
	PromptClass,
	Separator,
	TaskType,
	BitPrecision,
	SampleIdentifier,
	Quantization,
	Model
)

from eval_converters.common.adapter import BaseEvaluationAdapter, SupportedLibrary
from eval_converters.common.error import AdapterError
from eval_converters.common.utils import infer_quantization, infer_generation_args_default_values
from eval_converters.inspect.utils import detect_prompt_class#, get_adapter_class_from_method_string
from transformers import AutoConfig

class InspectAIAdapter(BaseEvaluationAdapter):
    """
    Adapter for transforming evaluation outputs from the Inspect AI library
    into the unified schema format.
    """
    SCENARIO_STATE_FILE = 'scenario_state.json'
    RUN_SPEC_FILE = 'run_spec.json'
    SCENARIO_FILE = 'scenario.json'

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.INSPECT_AI
        
    def transform_from_directory(self, dir_path: Union[str, Path]):
        raise NotImplementedError("Inspect AI adapter do not support loading logs from directory!")

    def transform_from_file(self, file_path: Union[str, Path]) -> Union[EvaluationResult, List[EvaluationResult]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File path {file_path} does not exists!')
        
        try:
            file_path = Path(file_path)
            eval_log: EvalLog = self._load_file(file_path)
            return self.transform(eval_log)
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

    def _transform_single(self, raw_data: EvalLog) -> List[EvaluationResult]:
        eval_spec: EvalSpec = raw_data.eval
        # 1. Model
        # 1.1. ModelInfo
        
        model_info = ModelInfo(
            name=eval_spec.model
        )

        # 1.2. InferenceSettings
        precision, quant_method, quant_type = infer_quantization(eval_spec.model)
        quantization = Quantization(
            bit_precision=precision,
            quantization_method=quant_method,
            quantization_type=quant_type
        )

        generate_config: GenerateConfig = eval_spec.model_generate_config
        generation_args_default: GenerationArgs = infer_generation_args_default_values(eval_spec.model)

        inference_settings = InferenceSettings(
            quantization=quantization,
            generation_args=GenerationArgs(
                temperature=generate_config.temperature or generation_args_default.temperature,
                top_p=generate_config.top_p or generation_args_default.top_p,
                top_k=generate_config.top_k or generation_args_default.top_k,
                max_tokens=generate_config.max_tokens or generation_args_default.max_tokens,
                stop_sequences=generate_config.stop_seqs or generation_args_default.stop_sequences,
                seed=generate_config.seed or generation_args_default.seed,
                frequency_penalty=generate_config.frequency_penalty or generation_args_default.frequency_penalty,
                presence_penalty=generate_config.presence_penalty or generation_args_default.presence_penalty,
                logit_bias=generate_config.logit_bias or generation_args_default.logit_bias,
                logprobs=generate_config.logprobs or generation_args_default.logprobs,
                top_logprobs=generate_config.top_logprobs or generation_args_default.top_logprobs
            )
        )

        model = Model(
            model_info=model_info,
            inference_settings=inference_settings,
        )

        # 2. PromptConfig
        # 2.1. PromptClass

        prompt_class = detect_prompt_class(raw_data.plan.steps[0].solver)
        prompt_config = PromptConfig(
            prompt_class=prompt_class,
            instruction_phrasing=InstructionPhrasing(
                name='template',
                text=raw_data.plan.steps[0].params.get('template', '')
            ),
            dimensions=None
        )

        if raw_data.results:
            metrics = raw_data.results.scores[0].metrics.keys()
            metrics = [m for m in metrics if 'stderr' not in m]
        elif eval_spec.scorers:
            metrics = [
                metric.name.split('/')[-1] for metric in eval_spec.scorers[0].metrics
                if 'stderr' not in metric.name 
            ]
        else:
            metrics = []

        evaluation_results: List[EvaluationResult] = []

        dataset = eval_spec.dataset
        samples = raw_data.samples

        for sample in samples:
            sample_identifier = SampleIdentifier(
                dataset_name=dataset.name,
                hf_repo=dataset.location,
                hf_index=sample.id,
                hf_split=None
            )

            # 3.2. ClassificationFields (required for classification tasks)
            classification_fields = {}
            if prompt_class == PromptClass.MultipleChoice: 
                classification_fields = {
                    "full_input": sample.messages[0].content,
                    "question": sample.input,
                    "choices": sample.choices,
                    "ground_truth": sample.target,
                }

            instance = Instance(
                task_type=TaskType.classification if prompt_class == PromptClass.MultipleChoice else TaskType.generation,
                raw_input=sample.input,
                language='en',  # FIXME: other languages?
                sample_identifier=sample_identifier,
                classification_fields=classification_fields,
            )
            
            # 4. Output
            if sample.scores:
                response = sample.scores.get('choice').answer
            else:
                response = sample.output.choices[0].message.content
            
            output = Output(
                response=sample.output.choices[0].message.content,
                explanation= sample.scores['choice'].explanation or sample.output.choices[0].message.content or None,
                generated_tokens_logprobs=sample.output.choices[0].logprobs
            )

            # 5. Evaluation
            metric_name = metrics[0] if metrics else None
            evaluation_method = EvaluationMethod(
                method_name=metric_name,
                description=f'{metric_name} is metric used for evaluation.'
            )

            score = self._get_score(
                response,
                sample.target,
                metric_name
            )

            evaluation = Evaluation(
                evaluation_method=evaluation_method,
                ground_truth=sample.target,
                score=score,
            )

            evaluation_id = f'inspect_ai_{str(eval_spec.eval_id)}_dataset_{sample_identifier.dataset_name}'
            evaluation_sample_id = f'inspect_ai_{str(eval_spec.eval_id)}_dataset_{sample_identifier.dataset_name}_id_{sample_identifier.hf_index}'

            evaluation_results.append(
                EvaluationResult(
                    schema_version=SCHEMA_VERSION,
                    evaluation_id=evaluation_id.replace('/', '_'),
                    evaluation_sample_id=evaluation_sample_id.replace('/', '_'),
                    model=model,
                    prompt_config=prompt_config,
                    instance=instance,
                    output=output,
                    evaluation=evaluation,
                )
            )

        return evaluation_results
        
    def _load_file(self, file_path) -> EvalLog:
        return read_eval_log(file_path)