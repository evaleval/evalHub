from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


class LMEvalRunner:         # noqa: D101
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        # Setup output directory at root level
        self.output_dir = Path(self.cfg.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_cli(self) -> List[str]:
        model = self.cfg.get("model", "hf")

        # Build model_args string from config
        model_args = self.cfg.get("model_args", "")
        if isinstance(model_args, dict):
            model_args = ",".join([f"{k}={v}" for k, v in model_args.items()])

        # Accept tasks as list or string
        raw_tasks = self.cfg.get("tasks", [])
        if isinstance(raw_tasks, (list, tuple)):
            tasks = ",".join(str(t) for t in raw_tasks)
        else:
            tasks = str(raw_tasks)

        batch_size = str(self.cfg.get("batch_size", 1))
        device = self.cfg.get("device", "cuda" if self.cfg.get("model") == "hf" else "cpu")

        cli = [
            "lm-eval",
            "--model", model,
            "--model_args", model_args,
            "--tasks", tasks,
            "--batch_size", batch_size,
            "--output_path", str(self.output_dir),
            "--device", device,
            "--log_samples",  # This creates predictions.jsonl
        ]

        # Add optional parameters
        if self.cfg.get("num_fewshot") is not None:
            cli.extend(["--num_fewshot", str(self.cfg["num_fewshot"])])

        if self.cfg.get("limit"):
            cli.extend(["--limit", str(self.cfg["limit"])])

        if self.cfg.get("temperature"):
            cli.extend(["--gen_kwargs", f"temperature={self.cfg['temperature']}"])

        if self.cfg.get("apply_chat_template"):
            cli.append("--apply_chat_template")

        if self.cfg.get("seed"):
            cli.extend(["--seed", str(self.cfg["seed"])])

        return cli

    def run(self) -> None:          # noqa: D401
        cli = self._build_cli()
        proc = subprocess.run(cli, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"LMEval failed with exit code {proc.returncode}. Output: {proc.stdout}")
        print(proc.stdout)
        
        # Write config to model subdirectory where adapter expects it
        if "pretrained=" in str(self.cfg.get("model_args", "")):
            model_name = str(self.cfg["model_args"]).split("pretrained=")[1].split(",")[0]
            
            model_dir = Path(self.output_dir) / model_name
            if model_dir.exists():
                with open(model_dir / "config.yaml", "w") as f:
                    yaml.safe_dump(self.cfg, f)
