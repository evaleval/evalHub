from datetime import datetime
from enum import Enum, IntEnum
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


class FrameworkEnum(str, Enum):
    inspect = 'inspect_ai'
    lmeval = 'lm_eval' # so on...

class Framework(BaseModel):
    name: FrameworkEnum
    version: str

class ModelUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int
    input_tokens_cache_read: int # additional info available in inspect logs

class Metric(BaseModel):
    name: str
    value: float
    params: Dict[str, any]

class SampleInput(BaseModel):
    sample_id: str
    input: str
    choices: List[str] # for MCQA task, how to define it in more universal way?
    target: str

class SampleOutput(BaseModel):
    sample_id: str
    answer: str # ex. 'A'
    explanation: str # ex. 'ANSWER: A'
    usage: ModelUsage # per sample
    total_time: float # in seconds?
    metrics: List[Metric] # metric values per sample
    prompt: str

class Dataset(BaseModel):
    name: str
    samples_count: int
    samples: List[SampleInput]
    location: str
    shuffled: bool

class Task(BaseModel):
    name: str # it should be normalized across all benchmarks 
    dataset: Dataset 
    type: str # for example multi-choice

class Model(BaseModel):
    model_name: str # ex. gpt-4o
    provider: str # ex. openai
    model_args: Dict[str, any] # temperature, ..., model exact version (for example gpt-4o-mini-2024-07-18)
    usage: ModelUsage

class Template(BaseModel):
    template: str # template for prompt

class ExperimentConfig(BaseModel):
    # work in progress
    limit: int # number of samples used for evaluation
    epoch: int

class Score(BaseModel):
    name: str # ex. 'choice'
    metrics: List[Metric]

class Results(BaseModel):
    total_samples: int
    completed_samples: int
    results_per_samples: List[SampleOutput]
    scores: List[Score]

class EvalSchema(BaseModel):
    framework: Framework
    task: Task
    model: Model
    template: Template
    results: Results
    exp_config: ExperimentConfig
    run_id: str # ?
    eval_id: str # ?
    started_at: datetime
    completed_at: datetime
