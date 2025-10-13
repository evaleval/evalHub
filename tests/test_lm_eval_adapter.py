from pathlib import Path
import json
import textwrap
import yaml

import pytest

from eval_converters.lm_eval.adapter import LMEvalAdapter

def create_tmp_lm_eval_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mock lm-eval output files"""
    # Ensure the directory exists
    tmp_path.mkdir(parents=True, exist_ok=True)
    
    # config.yaml
    cfg = {
        "model": "hf-causal",
        "model_args": {"pretrained": "gpt2"},
        "tasks": ["hellaswag"],
        "temperature": 0.7,
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # results.json (task-level)
    (tmp_path / "results.json").write_text(json.dumps({
        "hellaswag": {
            "acc_norm": 0.75,
        }
    }), encoding="utf-8")

    # samples.jsonl - two examples
    preds = textwrap.dedent(
        """
        {"task": "hellaswag", "idx": 0, "input": "Q1", "choices": ["A", "B", "C", "D"], "label": 2, "prediction": 2, "correct": true}
        {"task": "hellaswag", "idx": 1, "input": "Q2", "choices": ["A", "B", "C", "D"], "label": 1, "prediction": 3, "correct": false}
        """
    ).strip()
    (tmp_path / "samples.jsonl").write_text(preds, encoding="utf-8")

    return tmp_path


@pytest.fixture
def tmp_lm_eval_dir(tmp_path: Path) -> Path:
    """Pytest fixture wrapper for create_tmp_lm_eval_dir"""
    return create_tmp_lm_eval_dir(tmp_path)


def test_transform_from_directory(tmp_lm_eval_dir: Path):
    adapter = LMEvalAdapter()
    results = adapter.transform_from_directory(tmp_lm_eval_dir)

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert r.schema_version
        assert r.model.model_info.name == "gpt2"
        assert r.instance.raw_input.startswith("Q")
        assert r.evaluation.score in {0.0, 1.0}


def main():

    tmp_dir = create_tmp_lm_eval_dir(Path("/tmp/test_lm_eval"))

    try:
        test_transform_from_directory(tmp_dir)
    except Exception as e:
        print(f"test_transform_from_directory: FAILED - {e}")
        return False
    
    # Test on real output if available
    real_output_dir = Path("test_outputs")
    if real_output_dir.exists():
        subdirs = [d for d in real_output_dir.iterdir() if d.is_dir()]
        if subdirs:
            try:
                adapter = LMEvalAdapter()
                results = list(adapter.transform_from_directory(subdirs[0]))
                
                if results:
                    sample = results[0]
                    print(f"      Model: {sample.model.model_info.name}")
                    print(f"      Family: {sample.model.model_info.family}")
                    print(f"      Score: {sample.evaluation.score}")
                    print(f"      Method: {sample.evaluation.evaluation_method.method_name}")
            except Exception as e:
                print(f"Real output test failed: {e}")
        else:
            print("No output subdirectories found")
    else:
        print("No real output found (run LMEvalRunner first)")
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)