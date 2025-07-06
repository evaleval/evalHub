from typing import Any

from schema.eval_types import EvaluationResult
from common.adapter import BaseEvaluationAdapter, AdapterMetadata, SupportedLibrary

class HELMAdapter(BaseEvaluationAdapter):
    """
    Adapter for transforming evaluation outputs from the HELM library
    into the unified schema format.
    """

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="HELMAdapter",
            version="0.0.1",
            supported_library_versions=["0.5.6"],
            description="Adapter for transforming HELM evaluation outputs to unified schema format"
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.HELM

    def _transform_single(self, raw_data: Any) -> EvaluationResult:
        # Transform HELM data to unified schema

        # TODO:
        return EvaluationResult(
            # Populate with transformed data
            schema_version="0.0.1",
        )
    
    
# if __name__ == "__main__":
#     adapter = HELMAdapter()
#     print(adapter.metadata)
#     print(adapter.supported_library)
    
#     # Example raw data (to be replaced with actual HELM output)
#     raw_data = {}
#     try:
#         result = adapter._transform_single(raw_data)
#         print(result)
#     except Exception as e:
#         print(f"Error transforming data: {e}")