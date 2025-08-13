from __future__ import annotations

import subprocess
import sys
from pydantic_core.core_schema import model_schema
import yaml
from pathlib import Path
from typing import Any, Dict, List

class LMEvalRunner:         # noqa: D101
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)
        
        self.output_dir = Path(self.cfg.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_cli(self) -> List[str]:
        model = self.cfg.get("model", "hf")
        
        # Build model_args string from config
        model_args = self.cfg.get("model_args", "")
        if isinstance(model_args, dict):
            model_args = ",".join([f"{k}={v}" for k, v in model_args.items()])
        
        tasks = ",".join(self.cfg.get("tasks", []))
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

