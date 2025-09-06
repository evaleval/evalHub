from typing import Dict
from schema.eval_types import PromptClass

MULTIPLE_CHOICE_TASKS = {
    "hellaswag",
    "piqa",
    "siqa",
    "winogrande",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "copa",
    "wic",
    "anli_r1",
    "anli_r2",
    "anli_r3",
}

MAIN_METRIC_BY_TASK: Dict[str, str] = {
    **{t: "acc_norm" for t in ["hellaswag", "copa", "arc_easy", "arc_challenge"]},
    **{t: "acc" for t in ["piqa", "boolq", "winogrande", "openbookqa", "wic"]},
    # generative tasks often expose `exact_match` / `bleu` - handled ad-hoc
}

def detect_prompt_class(task_name: str) -> PromptClass:
    name = task_name.lower()
    if name in MULTIPLE_CHOICE_TASKS:
        return PromptClass.MultipleChoice
    return PromptClass.OpenEnded
    
