from schema.eval_types import PromptClass

def detect_prompt_class(adaptation_method: str) -> PromptClass:
    """
    Detect the PromptClass based on the adaptation method.
    """
    if 'multiple_choice' in adaptation_method.lower():
        return PromptClass.MultipleChoice
    
    return PromptClass.Completion # FIXME: how to deal with OpenEnded?