import os
import wget
import json
from typing import List, Dict, Sequence, Optional, Any
import tempfile
from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.presentation.schema import Schema, read_schema
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.benchmark.model_deployment_registry import get_model_deployment
from helm.benchmark.run_spec import RunSpec
from helm.common.codec import from_json
from dacite import from_dict
from pathlib import Path

from schema.eval_types import EvaluationResult, ModelInfo, Configuration, InferenceSettings, GenerationArgs, Quantization, BitPrecision, Method, Model, PromptConfig, Instance, Output, Evaluation, TaskType, PromptClass, SampleIdentifier, EvaluationMethod
from schema import SCHEMA_VERSION

from eval_converters.common.adapter import BaseEvaluationAdapter, AdapterMetadata, SupportedLibrary
from eval_converters.common.utils import detect_family, detect_hf_split, infer_quantization, extract_context_window_from_config
from .utils import detect_prompt_class, get_adapter_class_from_method_string

from transformers import AutoConfig

# run this just once in your process to initialize the registry
register_builtin_configs_from_helm_package()


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

	# to get the instance scores, we need to load the per_instance_stats.json file
	# and extract the main metric name from the schema
	# then, we can use the get_instance_scores_from_run_path method to get the instance scores
	@staticmethod
	def get_instance_scores(run_path: str, main_metric_name: str) -> Dict[str, float]:
		instance_scores: Dict[str, float] = {}
		per_instance_stats_path = os.path.join(run_path, "per_instance_stats.json")
		if os.path.exists(per_instance_stats_path):
			with open(per_instance_stats_path, "r") as f:
				per_instance_stats = from_json(f.read(), List[PerInstanceStats])
			for per_instance_stat in per_instance_stats:
				for stat in per_instance_stat.stats:
					if stat.name.name == main_metric_name:
						assert stat.mean is not None
						instance_scores[per_instance_stat.instance_id] = stat.mean
			return instance_scores

	@staticmethod
	def get_main_metric_name(run_path: str, schema_path: str) -> str:
		if schema_path.endswith(".json"):
			with open(schema_path, "r") as f:
				schema = from_json(f.read(), Schema)
		elif schema_path.endswith(".yaml"):
			schema = read_schema(schema_path)
		else:
			raise Exception(f"schema_path ended with unknown extension: {schema_path}")
		run_spec_path = os.path.join(run_path, "run_spec.json")
		with open(run_spec_path, "r") as f:
			run_spec = from_json(f.read(), RunSpec)
		for group in run_spec.groups:
			if group in schema.name_to_run_group and "main_name" in schema.name_to_run_group[group].environment:
				return schema.name_to_run_group[group].environment["main_name"]
		raise Exception(f"Could not find main metric name for {run_path}")
	
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
		# HELM does not provide context window size, try loading it from model deployment, else set to 1
		deployment = get_model_deployment(adapter_spec.model_deployment)
		if deployment and deployment.max_sequence_length is not None:
			context_window = deployment.max_sequence_length

		# if not available, try loading it from model config, else set to 1
		else:
			context_window = extract_context_window_from_config(adapter_spec.model)

		configuration = Configuration(
			context_window=context_window,
		)

		# 1.3. InferenceSettings
		try:
			precision, method = infer_quantization(adapter_spec.model)
		except Exception as e:
			print(f"Error getting quantization: {e}")
			precision = BitPrecision.none
			method = Method.None_

		quantization = Quantization(
			bit_precision=precision,
			method=method,
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

			try:
				# check for schema_*.yaml file in dir_path, if not found, download schema_*.yaml file from Github to dir_path using wget
				schema_path = os.path.join(dir_path, "schema_capabilities.yaml")
				if not os.path.exists(schema_path):
					wget.download("https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/benchmark/static/schema_capabilities.yaml", schema_path)

				main_metric_name = self.get_main_metric_name(dir_path, schema_path)
				instance_scores = self.get_instance_scores(dir_path, main_metric_name)
				score = instance_scores[request_state.instance.id]
				
			except Exception as e:
				print(f"Error getting instance scores: {e}")
				score = 0.0
			
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
	
	def _transform_single(self, raw_data, base_dir=None):
		"""
		Args:
			raw_data: Single evaluation record in HELM format (dict, JSON string, or file path)

		Returns:
			EvaluationResult in unified schema format
		"""
		# check if raw_data is a dictionary, JSON string, or file path
		if isinstance(raw_data, dict):
			data = raw_data
		elif isinstance(raw_data, (str, bytes)) and (raw_data.strip().startswith('{') or raw_data.strip().startswith('[')):
			# It's a JSON string
			data = json.loads(raw_data)
		else:
			# Assume it's a file path
			with open(raw_data, 'r') as f:
				data = json.load(f)

		
		scenario_state_dict = data['scenario_state_dict']
		run_spec_dict = data['run_spec_dict']
		scenario_dict = data['scenario_dict']

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
		# HELM does not provide context window size, try loading it from model deployment
		deployment = get_model_deployment(adapter_spec.model_deployment)
		if deployment and deployment.max_sequence_length is not None:
			context_window = deployment.max_sequence_length

		# if not available, try loading it from model config, else set to 1
		else:
			context_window = extract_context_window_from_config(adapter_spec.model)

		configuration = Configuration(
			context_window=context_window,
		)

		# 1.3. InferenceSettings
		try:
			precision, method = infer_quantization(adapter_spec.model)
		except Exception as e:
			print(f"Error getting quantization: {e}")
			precision = BitPrecision.none
			method = Method.None_

		quantization = Quantization(
			bit_precision=precision,
			method=method,
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

			try:
				schema_path = os.path.join(base_dir, "schema_capabilities.yaml")
				if not os.path.exists(schema_path):
					wget.download("https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/benchmark/static/schema_capabilities.yaml", schema_path)

				schema = read_schema(schema_path)
				main_metric_name = None

				# find the main metric name from the schema
				for group in run_spec.groups:
					if group in schema.name_to_run_group and "main_name" in schema.name_to_run_group[group].environment:
						main_metric_name = schema.name_to_run_group[group].environment["main_name"]
						break
				if main_metric_name is None:
					raise Exception("Could not find main metric name")

				# get the per instance stats from the data
				per_instance_stats = []
				if "per_instance_stats" in data:
					per_instance_stats = from_json(data["per_instance_stats"], List[PerInstanceStats])

				# get the instance scores from the per instance stats
				instance_scores = {}
				for per_instance_stat in per_instance_stats:
					for stat in per_instance_stat.stats:	
						if stat.name.name == main_metric_name:
							assert stat.mean is not None
							instance_scores[per_instance_stat.instance_id] = stat.mean
							break

			except Exception as e:
				print(f"Error getting instance scores: {e}")
				instance_scores = {}

			score = instance_scores.get(request_state.instance.id, 0.0)

			# 6. EvaluationResult
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
	