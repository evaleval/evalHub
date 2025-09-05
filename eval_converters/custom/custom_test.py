# eval_converters/custom/custom_test.py
import shutil, json
from pathlib import Path
from eval_converters.custom.adapter import CustomAdapter

# Clean & recreate the folder so no stale files linger
eval_dir = Path("tmp_custom_eval")
if eval_dir.exists():
    shutil.rmtree(eval_dir)
eval_dir.mkdir(parents=True, exist_ok=True)

single_data = {
    "evaluation_id": "llama-3.1-8b-test-single",
    "model": {"name": "meta-llama/Llama-3.1-8B-evals", "version": "1.0.0"},
    "prompt_config": {"prompt_class": "Completion"},  # <-- valid
    "instance": {
        "id": "ex1",
        "raw_input": "Translate 'hello' to French.",
        "language": "en",
        "sample_identifier": {
            "dataset_name": "dummy_dataset",
            "hf_repo": "meta-llama/Llama-3.1-8B-evals",
            "hf_split": "test",
            "hf_index": 0
        },
        "task_type": "generation"
    },
    "output": {"response": "Bonjour"}
}

list_data = [
    {
        "evaluation_id": "llama-3.1-8b-test-1",
        "model": {"name": "meta-llama/Llama-3.1-8B-evals", "version": "1.0.0"},
        "prompt_config": {"prompt_class": "Completion"},  # <-- valid
        "instance": {
            "id": "ex2",
            "raw_input": "Translate 'goodbye' to French.",
            "language": "en",
            "sample_identifier": {
                "dataset_name": "dummy_dataset",
                "hf_repo": "meta-llama/Llama-3.1-8B-evals",
                "hf_split": "test",
                "hf_index": 1
            },
            "task_type": "generation"
        },
        "output": {"response": "Au revoir"}
    },
    {
        "evaluation_id": "llama-3.1-8b-test-2",
        "model": {"name": "meta-llama/Llama-3.1-8B-evals", "version": "1.0.0"},
        "prompt_config": {"prompt_class": "Completion"},  # <-- valid
        "instance": {
            "id": "ex3",
            "raw_input": "Translate 'thank you' to French.",
            "language": "en",
            "sample_identifier": {
                "dataset_name": "dummy_dataset",
                "hf_repo": "meta-llama/Llama-3.1-8B-evals",
                "hf_split": "test",
                "hf_index": 2
            },
            "task_type": "generation"
        },
        "output": {"response": "Merci"}
    }
]

(eval_dir / "single.json").write_text(json.dumps(single_data, indent=2), encoding="utf-8")
(eval_dir / "list.json").write_text(json.dumps(list_data, indent=2), encoding="utf-8")

class DummyTool:
    name = "dummy_eval"
    version = "0.1"
    description = "Returns a fixed score"

    def evaluate(self, raw_data, **kwargs):
        return {"score": 0.92, "ground_truth": "dummy_gt"}

def main():
    adapter = CustomAdapter()
    adapter.set_tool(DummyTool())
    results = adapter.transform_from_directory(eval_dir)
    for r in results:
        print(f"ID={r.evaluation_id} | Model={r.model.model_info.name} | "
              f"Score={r.evaluation.score} | Response={r.output.response}")

if __name__ == "__main__":
    main()
