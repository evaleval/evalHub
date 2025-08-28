from schema.eval_types import PromptClass

def detect_prompt_class(adaptation_method: str) -> PromptClass:
    """
    Detect the PromptClass based on the adaptation method.
    """
    # builtin_inspect_solvers = [
    #     'prompt_template',
    #     'system_message',
    #     'user_message',
    #     'chain_of_thought',
    #     'generate',
    #     'self_critique',
    #     'multiple_choice'
    # ]
    if 'multiple_choice' in adaptation_method.lower():
        return PromptClass.MultipleChoice
    
    return PromptClass.Completion # FIXME: how to deal with OpenEnded?