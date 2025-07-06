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

class Study(BaseModel):
    study_id: str
    name: str
    description: str
    experiments: List[Experiment]


fixed_timestamp = datetime(2025, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


f1_metric = Metric(
    metric_id="m-001",
    name="F1 Score",
    description="Harmonic mean of precision and recall",
    parameters={"beta": 1}
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
                other_tags={"optimizer":"Adam"}
            ),
            optimization_metadata=OptimizationMetadata(
                type="quantization",
                bitwidth=8,
                algorithm="post-training",
                calibration_dataset="wiki_text"
            ),
            system_metadata=MachineInfo(
                cuda_version="11.7",
                torch_version="2.0.1",
                num_gpus=4,
                instance_size="400mb",
                other_system_info={"os":"Ubuntu 22.04"}
            ),
            evals_selected=[
                Eval(
                    eval_id="eval-squad",
                    dataset="SQuADv2",
                    pre_processing_recipe="tokenize",
                    post_processing_recipe="span_adjust",
                    metric=f1_metric
                )
            ],
            evaluation_query_metadata=[
                EvaluationQueryMetadata(
                    query_id="q-0001",
                    timestamp=fixed_timestamp,
                    input={"question":"What is X?"},
                    output={"answer":"Y"},
                    experiment_metadata_ref="exp-123"
                )
            ]
        )
    ]
)

json_output = study.model_dump_json(indent=2, by_alias=True, exclude_none=True)
data = json.loads(json_output)
save_json(data, "metadata.json")
