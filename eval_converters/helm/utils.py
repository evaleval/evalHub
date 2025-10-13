from schema.eval_types import PromptClass
from helm.benchmark.adaptation.adapters.adapter import Adapter
from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_EHR_INSTRUCTION,
    ADAPT_GENERATION,
    ADAPT_CHAT,
    ADAPT_GENERATION_MULTIMODAL,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT,
    ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_RANKING_BINARY,
)
from helm.benchmark.adaptation.adapters.binary_ranking_adapter import BinaryRankingAdapter
from helm.benchmark.adaptation.adapters.generation_adapter import GenerationAdapter
from helm.benchmark.adaptation.adapters.chat_adapter import ChatAdapter
from helm.benchmark.adaptation.adapters.language_modeling_adapter import LanguageModelingAdapter
from helm.benchmark.adaptation.adapters.multimodal.generation_multimodal_adapter import GenerationMultimodalAdapter
from helm.benchmark.adaptation.adapters.multimodal.multiple_choice_joint_multimodal_adapter import (
    MultipleChoiceJointMultimodalAdapter,
)
from helm.benchmark.adaptation.adapters.multiple_choice_calibrated_adapter import MultipleChoiceCalibratedAdapter
from helm.benchmark.adaptation.adapters.multiple_choice_joint_adapter import MultipleChoiceJointAdapter
from helm.benchmark.adaptation.adapters.multiple_choice_joint_chain_of_thought_adapter import (
    MultipleChoiceJointChainOfThoughtAdapter,
)
from helm.benchmark.adaptation.adapters.multiple_choice_separate_adapter import MultipleChoiceSeparateAdapter
from helm.benchmark.adaptation.adapters.ehr_instruction_adapter import EHRInstructionAdapter

def detect_prompt_class(adaptation_method: str) -> PromptClass:
    """
    Detect the PromptClass based on the adaptation method.
    """
    if 'multiple_choice' in adaptation_method.lower():
        return PromptClass.MultipleChoice
    
    return PromptClass.Completion # FIXME: how to deal with OpenEnded?

def get_adapter_class_from_method_string(method_str: str) -> type[Adapter]:
    method_str = method_str.strip().lower()

    mapping = {
        ADAPT_EHR_INSTRUCTION: EHRInstructionAdapter,
        ADAPT_GENERATION: GenerationAdapter,
        ADAPT_CHAT: ChatAdapter,
        ADAPT_LANGUAGE_MODELING: LanguageModelingAdapter,
        ADAPT_MULTIPLE_CHOICE_JOINT: MultipleChoiceJointAdapter,
        ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT: MultipleChoiceJointChainOfThoughtAdapter,
        ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL: MultipleChoiceSeparateAdapter,
        ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED: MultipleChoiceCalibratedAdapter,
        ADAPT_RANKING_BINARY: BinaryRankingAdapter,
        ADAPT_GENERATION_MULTIMODAL: GenerationMultimodalAdapter,
        ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL: MultipleChoiceJointMultimodalAdapter,
    }

    for key in mapping:
        if key in method_str:
            return mapping[key]

    raise ValueError(f"Unknown adapter method string: {method_str}")