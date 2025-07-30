import pytest
from pathlib import Path
from eval_tools.helm.adapter import HELMAdapter

@pytest.fixture
def adapter():
    return HELMAdapter()

def test_metadata_and_supported_library(adapter):
    metadata = adapter.metadata
    assert metadata.name == "HELMAdapter"
    assert metadata.version == "0.0.1"
    assert "0.5.6" in metadata.supported_library_versions
    assert "HELM evaluation outputs" in metadata.description

    supported_lib = adapter.supported_library
    assert supported_lib.name == "HELM"

def test_transform_from_directory(adapter):
    test_dir = Path(__file__).parent.resolve()
    output_dir_path = test_dir / 'data/helm/commonsense:dataset=hellaswag,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0'
    
    results = adapter.transform_from_directory(output_dir_path)
    
    assert isinstance(results, list)
    assert all(hasattr(r, 'schema_version') for r in results)
    assert all(r.model.model_info.name for r in results)
    assert all(r.instance.raw_input for r in results)
    assert len(results) > 0, "No results found in the output directory"
