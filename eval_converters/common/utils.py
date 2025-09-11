from schema.eval_types import BitPrecision, GenerationArgs, HfSplit, QuantizationMethod, QuantizationType
from transformers import AutoConfig, GenerationConfig


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

def infer_quantization_method_from_model_name(model_name_or_path: str) -> QuantizationMethod:
    for q_method in QuantizationMethod:
        if q_method.lower() in model_name_or_path.lower():
            return QuantizationMethod(q_method)
    
    if any(q_method in model_name_or_path for q_method in ["4bit", "8bit", "bitsandbytes", "bnb"]):
        return QuantizationMethod.BitsAndBytes

    return QuantizationMethod.none

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

    quantization_type = infer_quantization_method_from_model_name(model_name_or_path)

    method_map = {
        "gptq": QuantizationMethod.static,
        "gguf": QuantizationMethod.static,
        "awq": QuantizationMethod.static,
        "bitsandbytes": QuantizationMethod.dynamic,
        "quanto": QuantizationMethod.static,
        "hqq": QuantizationMethod.static,
        "torchao": QuantizationMethod.static,
        "ptq": QuantizationMethod.static,
        "smoothquant": QuantizationMethod.static,
        "qat": QuantizationMethod.static
    }

    method = method_map.get(quantization_type, QuantizationMethod.none)
    return precision, quantization_type, method

def infer_generation_args_default_values(model_name_or_path: str) -> GenerationArgs:
    try:
        gen_config = GenerationConfig.from_pretrained(model_name_or_path)
        return GenerationArgs(
            temperature=gen_config.temperature,
            top_p = gen_config.top_p,
            top_k = gen_config.top_k,
            max_tokens = gen_config.max_new_tokens,
            stop_sequences = gen_config.stop_strings,
            frequency_penalty = gen_config.repetition_penalty,
            presence_penalty = gen_config.diversity_penalty,
            logprobs = gen_config.output_logits
        )
    except Exception as e:
        return GenerationArgs()