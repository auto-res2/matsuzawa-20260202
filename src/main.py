import os
import subprocess
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


def apply_mode(cfg: DictConfig) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "run") and hasattr(cfg.run, "optuna"):
            cfg.run.optuna.n_trials = 0
            cfg.run.optuna.enabled = False
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    apply_mode(cfg)

    run_choice = cfg.run.run_id
    overrides = [
        f"runs@run={run_choice}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
        f"wandb.mode={cfg.wandb.mode}",
    ]

    if cfg.mode == "trial":
        overrides.extend(
            [
                "run.optuna.n_trials=0",
                "run.optuna.enabled=false",
            ]
        )

    cmd = [sys.executable, "-m", "src.train"] + overrides
    subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
