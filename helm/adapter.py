from typing import List
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.run_spec import RunSpec
from dacite import from_dict
from pathlib import Path

from schema.eval_types import EvaluationResult, ModelInfo, Configuration, InferenceSettings, GenerationArgs, Quantization, BitPrecision, Method, Model, PromptConfig, Instance, Output, Evaluation, TaskType, PromptClass, SampleIdentifier, EvaluationMethod
from schema import SCHEMA_VERSION

from common.adapter import BaseEvaluationAdapter, AdapterMetadata, SupportedLibrary
from common.utils import detect_family, detect_hf_split
from helm.utils import detect_prompt_class, get_adapter_class_from_method_string

class HELMAdapter(BaseEvaluationAdapter):
    """
    Adapter for transforming evaluation outputs from the HELM library
    into the unified schema format.
    """
    SCENARIO_STATE_FILE = 'scenario_state.json'
    RUN_SPEC_FILE = 'run_spec.json'
    SCENARIO_FILE = 'scenario.json'

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="HELMAdapter",
            version="0.0.1",
            supported_library_versions=["0.5.6"],
            description="Adapter for transforming HELM evaluation outputs to unified schema format"
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.HELM
    
    def transform_from_directory(self, dir_path):
        super().transform_from_directory(dir_path)
        
        scenario_state_dict = self._load_file(Path(f'{dir_path}/{self.SCENARIO_STATE_FILE}'))
        run_spec_dict = self._load_file(Path(f'{dir_path}/{self.RUN_SPEC_FILE}'))
        scenario_dict = self._load_file(Path(f'{dir_path}/{self.SCENARIO_FILE}')) # We don't load into Scenario instance as it is an abstract class

        # Load raw data object into a ScenarioState
        scenario_state = from_dict(data_class=ScenarioState, data=scenario_state_dict)
        adapter_spec = scenario_state.adapter_spec

        # Load raw data object into a RunSpec
        run_spec = from_dict(data_class=RunSpec, data=run_spec_dict)

        # Construct the EvaluationResult components
        # 1. Model
        # 1.1. ModelInfo
        model_info = ModelInfo(
            name=adapter_spec.model,
            family=detect_family(adapter_spec.model),
        )
        # 1.2. Configuration
        configuration = Configuration(
            context_window=1, # FIXME: HELM does not provide context window size, so we set it to 1. A simple fix is to get it from tokenizer?
        )
        # 1.3. InferenceSettings
        quantization = Quantization( # FIXME: HELM does not provide quantization info, so we set it to None default
            bit_precision=BitPrecision.none,
            method=Method.None_,
        )
        inference_settings = InferenceSettings(
            quantization=quantization,
            generation_args=GenerationArgs(
                temperature=adapter_spec.temperature,
                stop_sequences=adapter_spec.stop_sequences,
            )
        )
        
        # 2. PromptConfig
        # 2.1. PromptClass
        prompt_class = detect_prompt_class(adapter_spec.method)

        evaluation_results: List[EvaluationResult] = []
        for request_state in scenario_state.request_states:
            # 3. Instance
            # 3.1. SampleIdentifier
            sample_identifier = SampleIdentifier(
                dataset_name=scenario_dict['name'],
                hf_repo="", # FIXME: use HF repo if available
                hf_split=detect_hf_split(request_state.instance.split),
                hf_index=-1,  # FIXME: use actual index if available
            )
            
            # Extract ground truth: the first correct reference
            # FIXME: need to modify the schema to support evaluation with more than one ground truth: https://crfm-helm.readthedocs.io/en/latest/code/#adding-new-scenarios
            references = request_state.instance.references
            ground_truth = {}
            for i, ref in enumerate(references):
                if "correct" in ref.tags:
                    ground_truth = {
                        "id": str(i),
                        "text": ref.output.text,
                    }
                    break
            
            # 3.2. ClassificationFields (required for classification tasks)
            classification_fields = {}
            output_mapping_dict = request_state.output_mapping or {}
            if prompt_class == PromptClass.MultipleChoice: 
                choices = [{"id": k, "text": v} for k, v in output_mapping_dict.items()]

                classification_fields = {
                    "full_input": request_state.request.prompt,
                    "question": request_state.instance.input.text,
                    "choices": choices,
                    "ground_truth": ground_truth,
                }
            
            instance = Instance(
                task_type=TaskType.classification if prompt_class == PromptClass.MultipleChoice else TaskType.generation,
                raw_input=request_state.instance.input.text,
                language='en',  # FIXME: other languages?
                sample_identifier=sample_identifier,
                classification_fields=classification_fields,
            )
            
            # 4. Output
            output = Output(
                response=request_state.result.completions[0].text
            )

            # 5. Evaluation
            adapter = get_adapter_class_from_method_string(adapter_spec.method)
            evaluation_method = EvaluationMethod(
                method_name=adapter_spec.method,
                description=adapter.__class__.__doc__, # Use the adapter's docstring as description
            )
            score = 0.0 # TODO: implement scoring logic using HELM
            evaluation = Evaluation(
                evaluation_method=evaluation_method,
                ground_truth=ground_truth["text"],
                score=score,
            )
        
            evaluation_results.append(EvaluationResult(
                schema_version=SCHEMA_VERSION,
                evaluation_id=run_spec.name,
                model=Model(
                    model_info=model_info,
                    configuration=configuration,
                    inference_settings=inference_settings,
                ),
                prompt_config=PromptConfig(prompt_class=prompt_class),
                instance=instance,
                output=output,
                evaluation=evaluation,
            ))
        
        return evaluation_results
    
    def _transform_single(self, raw_data):
        # TODO:
        pass
    