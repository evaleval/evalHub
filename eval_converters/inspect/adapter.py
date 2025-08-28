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
	Configuration,
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
	QuantizationMethod,
    QuantizationType,
	Model
)

from eval_converters.common.adapter import BaseEvaluationAdapter, AdapterMetadata, SupportedLibrary
from eval_converters.common.error import AdapterError
from eval_converters.common.utils import detect_family, detect_hf_split, infer_quantization, extract_context_window_from_config
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
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="InspectAIAdapter",
            version="0.0.1",
            supported_library_versions=["0.3.112"],
            description="Adapter for transforming Inspect AI evaluation outputs to unified schema format"
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.INSPECT_AI
        
    def transform_from_directory(self, dir_path: Union[str, Path]):
        pass

    def transform_from_file(self, file_path: Union[str, Path]) -> Union[EvaluationResult, List[EvaluationResult]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File path {file_path} does not exists!')
        
        try:
            file_path = Path(file_path)
            eval_log: EvalLog = self._load_file(file_path)
            return self.transform(eval_log)
        except AdapterError as e:
            raise e # raise from e... fix it
        except Exception as e:
            raise AdapterError(f"Failed to load file {file_path}: {str(e)}")
    
    def _get_score(self, response: str, ground_truth: str, metric_name: str) -> float:
        if metric_name.strip().lower() == 'accuracy':
            score = float(response.strip().lower() == ground_truth.strip().lower())
            
            if not score:
                score = float(response.endswith(ground_truth))
            
            return score
        else:
            return 0.0

    def _transform_single(self, raw_data: EvalLog) -> List[EvaluationResult]:
        eval_spec: EvalSpec = raw_data.eval
        # 1. Model
        # 1.1. ModelInfo
        provider_name = eval_spec.model.rsplit('/', 1)[0]
        model_name = eval_spec.model.split('/')[-1] # raw_data.samples[0].output.model
        
        model_info = ModelInfo(
            name=model_name,
            family=detect_family(model_name),
            provider=provider_name
        )

        # 1.2. Configuration
        # architecture, parameters, is_instruct, hf_path - are these parameters even needed for evaluation uniqueness sake? It's hard to get them
        revision = eval_spec.revision
        commit_hash = revision.commit if revision else None
        context_window = extract_context_window_from_config(eval_spec.model)

        configuration = Configuration(
            context_window=context_window,
            revision=commit_hash
        )
        # 1.3. InferenceSettings
        precision, quant_method, quant_type = infer_quantization(eval_spec.model)
        quantization = Quantization(
            bit_precision=precision,
            method=quant_method,
            type=quant_type
            # info about quantization type = gguf, awq, so on...
        )

        generate_config: GenerateConfig = eval_spec.model_generate_config

        inference_settings = InferenceSettings(
            quantization=quantization,
            generation_args=GenerationArgs(
                temperature=generate_config.temperature or None,
                top_p=generate_config.top_p or None,
                top_k=generate_config.top_k or None,
                max_tokens=generate_config.max_tokens or None,
                stop_sequences=generate_config.stop_seqs or None,
                seed=generate_config.seed or None,
                frequency_penalty=generate_config.frequency_penalty or None,
                presence_penalty=generate_config.presence_penalty or None,
                logit_bias=generate_config.logit_bias or None,
                logprobs=generate_config.logprobs or None,
                top_logprobs=generate_config.top_logprobs or None
            )
        )

        model = Model(
            model_info=model_info,
            configuration=configuration,
            inference_settings=inference_settings,
            provider_name=provider_name
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

            evaluation_results.append(
                EvaluationResult(
                    schema_version=SCHEMA_VERSION,
                    evaluation_id=f'inspect/{str(eval_spec.eval_id)}',
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