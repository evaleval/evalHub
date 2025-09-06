from schema.eval_types import (
    BitPrecision, 
    Family, 
    HfSplit, 
    QuantizationMethod, 
    QuantizationType)
from transformers import AutoConfig

def detect_family(model_name: str) -> Family:
    """Return the Family enum if any of its values is a substring of model_name."""
    model_name_lower = model_name.lower()
    for family in Family:
        if family.value and family.value.lower() in model_name_lower:
            return family
    return Family.NoneType_None

def detect_hf_split(split_str: str) -> HfSplit:
    """
    Determines the type of dataset split from a given string.
    
    Args:
        split_str (str): The input string to classify.
        
    Returns:
        HfSplit: One of "train", "test", or "validation".
    """
    s = split_str.strip().lower()
    
    if s == "test":
        return HfSplit.test
    elif "train" in s:
        return HfSplit.train
    else:
        return HfSplit.validation

def infer_quantization_from_model_name(model_name_or_path: str) -> tuple[BitPrecision, QuantizationMethod, QuantizationType]:
    pass

def infer_quantization_from_model_config(model_name_or_path: str) -> tuple[BitPrecision, QuantizationMethod, QuantizationType]:
    pass

def infer_quantization(model_name_or_path: str) -> tuple[BitPrecision, QuantizationMethod, QuantizationType]:
    try:
        cfg = AutoConfig.from_pretrained(model_name_or_path)
    except Exception as e:
        return BitPrecision.none, QuantizationMethod.none, QuantizationType.none

    qcfg = getattr(cfg, 'quantization_config', None)
    if not qcfg:
        return BitPrecision.none, QuantizationMethod.none, QuantizationType.none

    bits = int(qcfg.get("bits") or qcfg.get("weight_bits") or qcfg.get("q_bits"))

    if bits == 8:
        precision = BitPrecision.int8
    elif bits == 4:
        precision = BitPrecision.int4
    elif bits == 16:
        precision = BitPrecision.float16
    elif bits == 32:
        precision = BitPrecision.float32
    else:
        precision = BitPrecision.none

    method_key = str(qcfg.get("quant_method") or "").lower()

    method_map = {
        "gptq": QuantizationMethod.gptq,
        "awq": QuantizationMethod.awq,
    }

    type_map = {
        "gptq": QuantizationType.static,
        "awq": QuantizationType.static,
        "bitsandbytes": QuantizationType.dynamic,
        "quanto": QuantizationType.static,
        "hqq": QuantizationType.static,
        "torchao": QuantizationType.static,
    }

    qmethod = method_map.get(method_key, QuantizationMethod.none)
    qtype = type_map.get(method_key, QuantizationType.none)
    return precision, qmethod, qtype

def extract_context_window_from_config(model):
    try:
        config = AutoConfig.from_pretrained(model)

        priority_fields = [
            "max_position_embeddings",
            "n_positions",
            "seq_len",
            "seq_length",
            "n_ctx",
            "sliding_window"
        ]

        context_window = next((getattr(config, f) for f in priority_fields if hasattr(config, f)), None)
        if context_window is None:
            context_window = 1
    
    except Exception as e:
        print(f"Error getting context window: {e}")
        context_window = 1
    
    finally:
        return context_window