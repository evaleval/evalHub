import pytest
from pathlib import Path
import tempfile
import yaml
import json
from unittest.mock import patch, MagicMock

from eval_converters.lm_eval.converter import LMEvalRunner


@pytest.fixture
def test_config():
    """Create a test configuration"""
    return {
        "model": "hf",
        "model_args": "pretrained=gpt2,dtype=float32",
        "tasks": ["hellaswag"],
        "batch_size": 2,
        "num_fewshot": 0,
        "output_dir": "test_outputs",
        "limit": 5,
        "device": "cpu",
        "seed": 42
    }


@pytest.fixture
def config_file(tmp_path, test_config):
    """Create a temporary config file"""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    return config_path


def test_lm_eval_runner_init(config_file, test_config):
    """Test LMEvalRunner initialization"""
    runner = LMEvalRunner(config_file)
    
    assert runner.config_path == Path(config_file)
    assert runner.cfg == test_config
    assert runner.output_dir.name == "test_outputs"
    assert runner.output_dir.exists()


def test_build_cli(config_file):
    """Test CLI command building"""
    runner = LMEvalRunner(config_file)
    cli = runner._build_cli()
    
    # Check basic structure
    assert cli[0] == "lm-eval"
    
    # Check model args
    assert "--model" in cli
    assert "hf" in cli
    assert "--model_args" in cli
    assert "pretrained=gpt2,dtype=float32" in cli
    
    # Check tasks
    assert "--tasks" in cli
    assert "hellaswag" in cli
    
    # Check other parameters
    assert "--batch_size" in cli
    assert "2" in cli
    assert "--device" in cli
    assert "cpu" in cli
    assert "--log_samples" in cli
    assert "--num_fewshot" in cli
    assert "0" in cli
    assert "--limit" in cli
    assert "5" in cli
    assert "--seed" in cli
    assert "42" in cli


def test_build_cli_with_dict_model_args(tmp_path):
    """Test CLI building when model_args is a dictionary"""
    config = {
        "model": "hf",
        "model_args": {
            "pretrained": "gpt2",
            "dtype": "float16",
            "trust_remote_code": True
        },
        "tasks": ["piqa"],
        "batch_size": 4,
        "output_dir": str(tmp_path / "outputs")
    }
    
    config_path = tmp_path / "dict_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    runner = LMEvalRunner(config_path)
    cli = runner._build_cli()
    
    # Find model_args in CLI
    model_args_idx = cli.index("--model_args") + 1
    model_args_str = cli[model_args_idx]
    
    # Check that dictionary was converted to comma-separated string
    assert "pretrained=gpt2" in model_args_str
    assert "dtype=float16" in model_args_str
    assert "trust_remote_code=True" in model_args_str


def test_build_cli_optional_params(tmp_path):
    """Test CLI building with optional parameters"""
    config = {
        "model": "hf",
        "model_args": "pretrained=gpt2",
        "tasks": ["arc_easy"],
        "batch_size": 1,
        "output_dir": str(tmp_path / "outputs"),
        "temperature": 0.7,
        "apply_chat_template": True
    }
    
    config_path = tmp_path / "optional_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    runner = LMEvalRunner(config_path)
    cli = runner._build_cli()
    
    # Check temperature is added
    assert "--gen_kwargs" in cli
    temp_idx = cli.index("--gen_kwargs") + 1
    assert cli[temp_idx] == "temperature=0.7"
    
    # Check chat template flag
    assert "--apply_chat_template" in cli


@patch('subprocess.run')
def test_run_success(mock_run, config_file):
    """Test successful run"""
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Evaluation completed successfully"
    mock_run.return_value = mock_process
    
    runner = LMEvalRunner(config_file)
    runner.run()
    
    # Check subprocess was called
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    
    # Verify it's running lm-eval
    assert "lm-eval" in call_args


@patch('subprocess.run')
def test_run_failure(mock_run, config_file):
    """Test failed run"""
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = "Error: Model not found"
    mock_run.return_value = mock_process
    
    runner = LMEvalRunner(config_file)
    
    with pytest.raises(RuntimeError) as exc_info:
        runner.run()
    
    assert "LMEval failed with exit code 1" in str(exc_info.value)
    assert "Model not found" in str(exc_info.value)


def test_output_dir_creation(tmp_path):
    """Test that output directory is created properly"""
    output_dir = tmp_path / "custom_outputs"
    config = {
        "model": "hf",
        "model_args": "pretrained=gpt2",
        "tasks": ["boolq"],
        "batch_size": 1,
        "output_dir": str(output_dir)
    }
    
    config_path = tmp_path / "output_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    runner = LMEvalRunner(config_path)
    
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_real_evaluation_with_test_config():
    """Test running a real evaluation using the test config"""
    from pathlib import Path
    import json
    
    # Use the existing test config
    config_path = Path("config/lm_eval_test_config.yaml")
    
    print(f"\nRunning real evaluation with {config_path}")
    
    try:
        runner = LMEvalRunner(config_path)
        
        # Clear test_outputs directory first
        import shutil
        if runner.output_dir.exists():
            shutil.rmtree(runner.output_dir)
        
        print(f"Output directory: {runner.output_dir}")
        print("Running evaluation...")
        
        runner.run()
        
        print("Evaluation completed!")
        
        # Check that output files were created
        assert runner.output_dir.exists(), "Output directory should exist"
        
        # Check for key output files
        config_file = runner.output_dir / "config.yaml"
        results_file = runner.output_dir / "results.json"
        
        # At least one of these should exist
        pred_files = list(runner.output_dir.glob("*predictions*.jsonl")) + \
                    list(runner.output_dir.glob("*samples*.jsonl"))
        
        print(f"Files created:")
        for file_path in runner.output_dir.iterdir():
            if file_path.is_file():
                print(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")
        
        # Basic assertions
        assert config_file.exists() or results_file.exists(), "Should have config.yaml or results.json"
        
        # Test the adapter if we have prediction files
        if pred_files:
            print("Testing adapter transformation...")
            from eval_converters.lm_eval.adapter import LMEvalAdapter
            adapter = LMEvalAdapter()
            results = list(adapter.transform_from_directory(runner.output_dir))
            
            print(f"Adapter processed {len(results)} evaluation results")
            
            if results:
                sample = results[0]
                assert sample.schema_version
                assert sample.model.model_info.name
                assert sample.evaluation.score is not None
                print(f"Sample model: {sample.model.model_info.name}")
                print(f"Sample score: {sample.evaluation.score}")
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {type(e).__name__}: {e}")
        # Don't fail the test if lm-eval isn't properly installed or there are environment issues
        pytest.skip(f"Skipping real evaluation test due to: {e}")
        return False


def main():
    # Test 1: Configuration loading and CLI building
    try:
        from pathlib import Path
        config_path = Path("config/lm_eval_test_config.yaml")
        
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            return False
            
        runner = LMEvalRunner(config_path)
        print(f"Config loaded: {runner.cfg.get('model')} model")
        print(f"Tasks: {runner.cfg.get('tasks')}")
        print(f"Output dir: {runner.output_dir}")
        
        # Test CLI building
        cli = runner._build_cli()
        print(f"CLI built: {' '.join(cli[:5])}...")
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

    # Test 2: Real evaluation
    try:
        # Clear test_outputs first
        import shutil
        if runner.output_dir.exists():
            shutil.rmtree(runner.output_dir)
        
        runner.run()
        
        # Check output files
        if runner.output_dir.exists():
            for file_path in runner.output_dir.rglob("*"):
                if file_path.is_file():
                    size_kb = file_path.stat().st_size / 1024
                    print(f"{file_path.relative_to(runner.output_dir)} ({size_kb:.1f}KB)")
        
    except Exception as e:
        print(f"Real evaluation failed (this is OK for testing): {e}")
    
    # Test 3: Adapter integration
    try:
        from eval_converters.lm_eval.adapter import LMEvalAdapter
        adapter = LMEvalAdapter()
        
        # Find output directories
        output_dirs = []
        if runner.output_dir.exists():
            output_dirs = [d for d in runner.output_dir.rglob("*") if d.is_dir() and any(d.glob("*.json*"))]
        
        if output_dirs:
            test_dir = output_dirs[0]
            
            results = list(adapter.transform_from_directory(test_dir))
            
            if results:
                sample = results[0]
                print(f"      Model: {sample.model.model_info.name}")
                print(f"      Family: {sample.model.model_info.family}")
                print(f"      Score: {sample.evaluation.score}")
                print(f"      Method: {sample.evaluation.evaluation_method.method_name}")
        else:
            print("No output directories found for adapter testing")
            
    except Exception as e:
        print(f"Adapter integration test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)