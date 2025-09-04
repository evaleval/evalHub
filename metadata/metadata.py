from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from utils import save_json
import json


class ModelCheckpointMetadata(BaseModel):
    checkpoint_name: str
    precision: float
    num_layers: int
    other_tags: Dict[str, Any] = Field(default_factory=dict)


class OptimizationMetadata(BaseModel):
    type: str
    bitwidth: int
    algorithm: str
    calibration_dataset: str


class MachineInfo(BaseModel):
    cuda_version: str
    torch_version: str
    num_gpus: int
    instance_size: str
    other_system_info: Dict[str, Any] = Field(default_factory=dict)


class Metric(BaseModel):
    metric_id: str
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Eval(BaseModel):
    eval_id: str
    dataset: str
    pre_processing_recipe: str
    post_processing_recipe: str
    metric: Metric


class EvaluationQueryMetadata(BaseModel):
    query_id: str
    timestamp: datetime
    input: Dict[str, Any]
    output: Dict[str, Any]
    experiment_metadata_ref: str


class Experiment(BaseModel):
    experiment_id: str
    model_checkpoint_metadata: ModelCheckpointMetadata
    optimization_metadata: OptimizationMetadata
    system_metadata: MachineInfo
    evals_selected: List[Eval]
    evaluation_query_metadata: List[EvaluationQueryMetadata]

    # New fields for experiment lineage and baseline tracking
    derived_from: Optional[str] = None  # ID of the parent experiment
    is_baseline: bool = False  # Whether this is a baseline experiment

    def experiment_signature(self) -> str:
        """
        Generate a unique signature for the experiment based on key metadata.

        Returns:
            str: A signature string combining model name, precision, and optimization details.
        """
        # Start with model checkpoint information
        model_name = self.model_checkpoint_metadata.checkpoint_name
        precision = int(self.model_checkpoint_metadata.precision)
        base_signature = f"{model_name}_fp{precision}"

        # Add optimization metadata if available
        if self.optimization_metadata:
            opt = self.optimization_metadata
            opt_signature = (
                f"_{opt.type}{opt.bitwidth}_{opt.algorithm}_{opt.calibration_dataset}"
            )
            base_signature += opt_signature

        return base_signature


class Study(BaseModel):
    study_id: str
    name: str
    description: str
    experiments: List[Experiment]


def compare_experiments(exp1: Experiment, exp2: Experiment) -> Dict[str, Any]:
    """
    Compare two experiments and return a structured diff showing which fields differ.

    Args:
        exp1: First experiment to compare
        exp2: Second experiment to compare

    Returns:
        Dict containing only the fields that differ between the experiments,
        with exp1 and exp2 values for each differing field.
    """
    diffs = {}

    # List of top-level fields to compare (excluding lists for now)
    fields_to_compare = [
        "experiment_id",
        "model_checkpoint_metadata",
        "optimization_metadata",
        "system_metadata",
        "derived_from",
        "is_baseline",
    ]

    for field_name in fields_to_compare:
        val1 = getattr(exp1, field_name)
        val2 = getattr(exp2, field_name)

        # For BaseModel objects, compare their dict representation
        if hasattr(val1, "model_dump") and hasattr(val2, "model_dump"):
            dict1 = val1.model_dump()
            dict2 = val2.model_dump()
            if dict1 != dict2:
                diffs[field_name] = {"exp1": dict1, "exp2": dict2}
        # For simple values, compare directly
        elif val1 != val2:
            diffs[field_name] = {"exp1": val1, "exp2": val2}

    # Handle list fields separately (evals and evaluation queries)
    if len(exp1.evals_selected) != len(exp2.evals_selected):
        diffs["evals_selected"] = {
            "exp1": [eval.model_dump() for eval in exp1.evals_selected],
            "exp2": [eval.model_dump() for eval in exp2.evals_selected],
        }
    else:
        # Compare individual evals if same length
        for i, (eval1, eval2) in enumerate(
            zip(exp1.evals_selected, exp2.evals_selected)
        ):
            if eval1.model_dump() != eval2.model_dump():
                diffs[f"evals_selected[{i}]"] = {
                    "exp1": eval1.model_dump(),
                    "exp2": eval2.model_dump(),
                }

    if len(exp1.evaluation_query_metadata) != len(exp2.evaluation_query_metadata):
        diffs["evaluation_query_metadata"] = {
            "exp1": [query.model_dump() for query in exp1.evaluation_query_metadata],
            "exp2": [query.model_dump() for query in exp2.evaluation_query_metadata],
        }

    return diffs


fixed_timestamp = datetime(2025, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


f1_metric = Metric(
    metric_id="m-001",
    name="F1 Score",
    description="Harmonic mean of precision and recall",
    parameters={"beta": 1},
)

study = Study(
    study_id="study-001",
    name="Quantization Study",
    description="Assessing 8-bit quantized BERT",
    experiments=[
        Experiment(
            experiment_id="exp-123",
            model_checkpoint_metadata=ModelCheckpointMetadata(
                checkpoint_name="bert-base-uncased",
                precision=16.0,
                num_layers=12,
                other_tags={"optimizer": "Adam"},
            ),
            optimization_metadata=OptimizationMetadata(
                type="quantization",
                bitwidth=8,
                algorithm="post-training",
                calibration_dataset="wiki_text",
            ),
            system_metadata=MachineInfo(
                cuda_version="11.7",
                torch_version="2.0.1",
                num_gpus=4,
                instance_size="400mb",
                other_system_info={"os": "Ubuntu 22.04"},
            ),
            evals_selected=[
                Eval(
                    eval_id="eval-squad",
                    dataset="SQuADv2",
                    pre_processing_recipe="tokenize",
                    post_processing_recipe="span_adjust",
                    metric=f1_metric,
                )
            ],
            evaluation_query_metadata=[
                EvaluationQueryMetadata(
                    query_id="q-0001",
                    timestamp=fixed_timestamp,
                    input={"question": "What is X?"},
                    output={"answer": "Y"},
                    experiment_metadata_ref="exp-123",
                )
            ],
        )
    ],
)

json_output = study.model_dump_json(indent=2, by_alias=True, exclude_none=True)
data = json.loads(json_output)
save_json(data, "metadata.json")


if __name__ == "__main__":
    # Create a Llama 3.1 baseline experiment
    llama_baseline = Experiment(
        experiment_id="llama-exp-001",
        model_checkpoint_metadata=ModelCheckpointMetadata(
            checkpoint_name="meta-llama/Llama-3.1-8B",
            precision=16.0,
            num_layers=32,
            other_tags={"optimizer": "AdamW", "architecture": "llama"},
        ),
        optimization_metadata=OptimizationMetadata(
            type="none", bitwidth=16, algorithm="baseline", calibration_dataset="none"
        ),
        system_metadata=MachineInfo(
            cuda_version="12.1",
            torch_version="2.1.0",
            num_gpus=4,
            instance_size="80GB",
            other_system_info={"os": "Ubuntu 22.04", "vram_per_gpu": "80GB"},
        ),
        evals_selected=[
            Eval(
                eval_id="eval-hellaswag-llama",
                dataset="HellaSwag",
                pre_processing_recipe="llama_format",
                post_processing_recipe="multiple_choice_extract",
                metric=Metric(
                    metric_id="m-002",
                    name="Accuracy",
                    description="Multiple choice accuracy",
                    parameters={},
                ),
            )
        ],
        evaluation_query_metadata=[
            EvaluationQueryMetadata(
                query_id="q-llama-001",
                timestamp=fixed_timestamp,
                input={"prompt": "Complete the following: The capital of France is"},
                output={"completion": "Paris"},
                experiment_metadata_ref="llama-exp-001",
            )
        ],
        is_baseline=True,
    )

    # Test the experiment signature
    print(f"Llama baseline signature: {llama_baseline.experiment_signature()}")

    # Create a quantized Llama 3.1 experiment derived from baseline
    llama_quantized = Experiment(
        experiment_id="llama-exp-002",
        model_checkpoint_metadata=ModelCheckpointMetadata(
            checkpoint_name="meta-llama/Llama-3.1-8B",
            precision=8.0,
            num_layers=32,
            other_tags={"optimizer": "AdamW", "architecture": "llama"},
        ),
        optimization_metadata=OptimizationMetadata(
            type="quantization", bitwidth=8, algorithm="QLoRA", calibration_dataset="c4"
        ),
        system_metadata=MachineInfo(
            cuda_version="12.1",
            torch_version="2.1.0",
            num_gpus=2,
            instance_size="40GB",
            other_system_info={"os": "Ubuntu 22.04", "vram_per_gpu": "40GB"},
        ),
        evals_selected=llama_baseline.evals_selected,
        evaluation_query_metadata=[
            EvaluationQueryMetadata(
                query_id="q-llama-002",
                timestamp=fixed_timestamp,
                input={"prompt": "Complete the following: The capital of France is"},
                output={"completion": "Paris"},
                experiment_metadata_ref="llama-exp-002",
            )
        ],
        derived_from="llama-exp-001",
        is_baseline=False,
    )

    print(f"Llama quantized signature: {llama_quantized.experiment_signature()}")

    # Compare the Llama experiments
    llama_diffs = compare_experiments(llama_baseline, llama_quantized)
    print("Differences between Llama 3.1 experiments:")
    for field, diff in llama_diffs.items():
        print(f"  {field}: exp1={diff['exp1']} vs exp2={diff['exp2']}")
