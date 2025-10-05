# eval_converters/custom/adapter.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Union

from eval_converters.common.adapter import BaseEvaluationAdapter, AdapterMetadata, SupportedLibrary, AdapterError
from eval_converters.common.utils import detect_family
from schema import SCHEMA_VERSION
from schema.eval_types import (
    EvaluationResult,
    ModelInfo,
    Model,
    PromptConfig,
    Instance,
    Output,
    Evaluation,
    EvaluationMethod,
    Configuration,
    InferenceSettings,
    GenerationArgs,
    Quantization,
    BitPrecision,
    Method,
    PromptClass,
)

@runtime_checkable
class EvaluationTool(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def description(self) -> str: ...

    def evaluate(self, raw_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Given one raw JSON object, return a metrics dict.
        Expected keys: 'score' (float), optional 'ground_truth' (str).
        """
        ...

class CustomAdapter(BaseEvaluationAdapter):
    """
    Adapter for transforming outputs from user-defined evaluation tools
    into the unified schema format.

    Usage:
        adapter = CustomAdapter().set_tool(my_tool)
        results = adapter.transform_from_directory("path/to/jsons")
        # or: adapter.transform_from_file("file.json")
        # or: adapter.transform(one_dict_or_list)
    """

    def __init__(self, tool: Optional[EvaluationTool] = None, strict_validation: bool = True):
        super().__init__(strict_validation=strict_validation)
        self._tool: Optional[EvaluationTool] = None
        if tool is not None:
            self.set_tool(tool)

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="CustomAdapter",
            version="0.0.1",
            supported_library_versions=["*"],
            description="Minimal adapter for transforming custom evaluation outputs to the unified schema",
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.CUSTOM

    def set_tool(self, tool: EvaluationTool) -> "CustomAdapter":
        """Attach the user-provided evaluation tool (implements EvaluationTool)."""
        if not isinstance(tool, EvaluationTool):
            # Soft check: allow typed objects even if isinstance fails
            # (runtime_checkable lets this work for proper protocols).
            missing = [a for a in ("name", "version", "description", "evaluate") if not hasattr(tool, a)]
            if missing:
                raise TypeError(f"Tool is missing required attributes: {missing}")
        self._tool = tool
        return self

    def _require_tool(self) -> EvaluationTool:
        tool = self._tool
        if tool is None:
            raise ValueError("No tool set. Call set_tool(tool) before transforming.")
        return tool

    def transform_from_directory(self, dir_path: Union[str, Path]) -> List[EvaluationResult]:
        """
        Reads all *.json and *.jsonl files (non-recursive) in dir_path and transforms them.
        Respects strict_validation: errors on individual files will either raise or log+skip.
        """
        dir_path = Path(dir_path)
        super().transform_from_directory(dir_path)

        results: List[EvaluationResult] = []
        # Support both JSON and JSONL
        for file_path in sorted(dir_path.glob("*.json")) + sorted(dir_path.glob("*.jsonl")):
            try:
                data = self._load_file(file_path)  # provided by BaseEvaluationAdapter
                transformed = self.transform(data)  # also provided by BaseEvaluationAdapter
                if isinstance(transformed, list):
                    results.extend(transformed)
                else:
                    results.append(transformed)
            except Exception as e:
                try:
                    self._handle_transformation_error(e, f"file {file_path}")
                except Exception:
                    raise
        return results

    def _transform_single(self, raw_data: Any) -> EvaluationResult:
        """
        Transform a single evaluation record in *custom tool* format.
        Accepts: dict, JSON string/bytes, or file path.
        """
        tool = self._require_tool()

        # normalize raw_data to a dict
        if isinstance(raw_data, dict):
            data: Dict[str, Any] = raw_data
        elif isinstance(raw_data, (str, bytes, bytearray)):
            # string JSON or bytes will parse; if it's a path string, let json.load fail
            try:
                s = raw_data.decode("utf-8") if isinstance(raw_data, (bytes, bytearray)) else raw_data
                s = s.strip()
                if s.startswith("{") or s.startswith("["):
                    parsed = json.loads(s)
                    if not isinstance(parsed, dict):
                        raise TypeError(f"_transform_single expected JSON object, got {type(parsed).__name__}")
                    data = parsed
                else:
                    # treat as file path like string
                    with open(s, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if not isinstance(loaded, dict):
                        raise TypeError(f"_transform_single expected JSON object from file, got {type(loaded).__name__}")
                    data = loaded
            except Exception as e:
                raise AdapterError(f"Failed to parse raw_data: {e}") from e
        else:
            # Any other object should assume a path like and try file load
            try:
                with open(raw_data, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise TypeError(f"_transform_single expected JSON object from file, got {type(loaded).__name__}")
                data = loaded
            except Exception as e:
                raise AdapterError(f"Failed to read from {raw_data}: {e}") from e

        # Build Model
        model_name = data.get("model", {}).get("name", "custom_tool")
        model_version = data.get("model", {}).get("version", "1.0.0")
        model_info = ModelInfo(
            name=model_name,
            version=model_version,
            family=detect_family(model_name),
        )
        configuration = Configuration(context_window=1)
        inference_settings = InferenceSettings(
            quantization=Quantization(bit_precision=BitPrecision.none, method=Method.None_),
            generation_args=GenerationArgs(),
        )
        model = Model(
            model_info=model_info,
            configuration=configuration,
            inference_settings=inference_settings,
        )

        # PromptConfig
        pc = (data.get("prompt_config") or {}).copy()
        if "prompt_class" not in pc:
            pc["prompt_class"] = next(iter(PromptClass))  # safe default
        prompt_config = PromptConfig(**pc)

        # Instance
        instance_payload = (data.get("instance") or {}).copy()
        instance_payload.setdefault("language", "en")
        instance_payload.setdefault(
            "sample_identifier",
            {"dataset_name": "custom_dataset", "hf_repo": "", "hf_split": "test", "hf_index": -1},
        )
        instance = Instance(**instance_payload) if isinstance(instance_payload, dict) else Instance()

        output_payload = data.get("output", {"response": ""})
        output = Output(**output_payload)

        metrics = tool.evaluate(data)
        if not isinstance(metrics, dict):
            raise TypeError(f"{tool.name}.evaluate() must return dict, got {type(metrics).__name__}")
        try:
            score = float(metrics.get("score", 0.0))
        except Exception as e:
            raise TypeError(f"{tool.name}.evaluate() returned non-numeric 'score': {metrics.get('score')!r}") from e

        evaluation_method = EvaluationMethod(
            method_name=tool.name,
            description=tool.description,
        )
        evaluation = Evaluation(
            evaluation_method=evaluation_method,
            ground_truth=metrics.get("ground_truth", ""),
            score=score,
        )

        return EvaluationResult(
            schema_version=SCHEMA_VERSION,
            evaluation_id=data.get("evaluation_id", ""),
            model=model,
            prompt_config=prompt_config,
            instance=instance,
            output=output,
            evaluation=evaluation,
        )
