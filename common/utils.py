from schema.eval_types import Family

def detect_family(model_name: str) -> Family:
    """Return the Family enum if any of its values is a substring of model_name."""
    model_name_lower = model_name.lower()
    for family in Family:
        if family.value and family.value.lower() in model_name_lower:
            return family
    return Family.NoneType_None