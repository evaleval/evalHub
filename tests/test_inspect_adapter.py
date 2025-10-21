import json
from pathlib import Path

from eval_converters.inspect.adapter import InspectAIAdapter
from schema.eval_types import (
    EvaluationLog, 
    EvaluatorRelationship,
    SourceData,
    SourceMetadata,

)

def _load_eval(adapter, filepath, source_metadata):
    eval_path = Path(filepath)
    converted_eval = adapter.transform_from_file(eval_path, source_metadata=source_metadata)
    assert isinstance(converted_eval, EvaluationLog)
    assert isinstance(converted_eval.source_data, SourceData)

    assert converted_eval.evaluation_source.evaluation_source_name == 'inspect_ai'
    assert converted_eval.evaluation_source.evaluation_source_type.value == 'evaluation_platform'

    return converted_eval


def test_pubmedqa_eval():
    adapter = InspectAIAdapter()
    source_metadata = SourceMetadata(
        source_organization_name='TestOrg',
        evaluator_relationship=EvaluatorRelationship.first_party,
    )

    converted_eval = _load_eval(adapter, 'tests/data/inspect/data_pubmedqa_gpt4o_mini.json', source_metadata)

    assert converted_eval.retrieved_timestamp == '1751553870.0'
    
    assert converted_eval.source_data.dataset_name == 'pubmed_qa'
    assert converted_eval.source_data.hf_repo == 'bigbio/pubmed_qa'
    assert len(converted_eval.source_data.sample_ids) == 2

    assert converted_eval.model_info.name == 'openai/azure/gpt-4o-mini-2024-07-18'
    assert converted_eval.model_info.id == 'openai/azure/gpt-4o-mini-2024-07-18'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'openai/azure'

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 1.0

    results_per_sample = converted_eval.detailed_evaluation_results_per_samples
    sample_ids = [sample.sample_id for sample in results_per_sample]
    assert sorted(sample_ids) == ['12377809', '26163474']
    assert results_per_sample[0].ground_truth == 'A'
    assert results_per_sample[0].response == 'A'
    assert results_per_sample[0].choices == ['yes', 'no', 'maybe']


def test_arc_sonnet_eval():
    adapter = InspectAIAdapter()
    source_metadata = SourceMetadata(
        source_organization_name='TestOrg',
        evaluator_relationship=EvaluatorRelationship.first_party,
    )

    converted_eval = _load_eval(adapter, 'tests/data/inspect/data_arc_sonnet.json', source_metadata)

    assert converted_eval.retrieved_timestamp == '1761000045.0'

    assert converted_eval.source_data.dataset_name == 'ai2_arc'
    assert converted_eval.source_data.hf_repo == 'allenai/ai2_arc'
    assert len(converted_eval.source_data.sample_ids) == 5

    assert converted_eval.model_info.name == 'anthropic/claude-sonnet-4-0'
    assert converted_eval.model_info.id == 'anthropic/claude-sonnet-4-0'
    assert converted_eval.model_info.developer == 'anthropic'
    assert converted_eval.model_info.inference_platform == 'anthropic'

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 1.0

    results_per_sample = converted_eval.detailed_evaluation_results_per_samples
    sample_ids = [sample.sample_id for sample in results_per_sample]
    assert sorted(sample_ids) == ['1', '2', '3', '4', '5']
    assert results_per_sample[0].ground_truth == 'A'
    assert results_per_sample[0].response == 'A'
    assert 'Sunlight is the source of energy for nearly all ecosystems.' in results_per_sample[0].choices


def test_arc_qwen_eval():
    adapter = InspectAIAdapter()
    source_metadata = SourceMetadata(
        source_organization_name='TestOrg',
        evaluator_relationship=EvaluatorRelationship.first_party,
    )

    converted_eval = _load_eval(adapter, 'tests/data/inspect/data_arc_qwen.json', source_metadata)

    assert converted_eval.retrieved_timestamp == '1761001924.0'

    assert converted_eval.source_data.dataset_name == 'ai2_arc'
    assert converted_eval.source_data.hf_repo == 'allenai/ai2_arc'
    assert len(converted_eval.source_data.sample_ids) == 3

    assert converted_eval.model_info.name == 'ollama/qwen2.5:0.5b'
    assert converted_eval.model_info.id == 'ollama/qwen2.5:0.5b'
    assert converted_eval.model_info.developer == 'ollama'
    assert converted_eval.model_info.inference_platform == 'ollama'

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 0.3333333333333333

    results_per_sample = converted_eval.detailed_evaluation_results_per_samples
    sample_ids = [sample.sample_id for sample in results_per_sample]
    assert sorted(sample_ids) == ['1', '2', '3']
    assert results_per_sample[1].ground_truth == 'B'
    assert results_per_sample[1].response == 'D'
    assert results_per_sample[1].choices == ["safety goggles", "breathing mask", "rubber gloves", "lead apron"]