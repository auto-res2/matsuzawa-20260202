import os
import random
from pathlib import Path
from typing import Dict, List, Any

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src import model as model_lib
from src import preprocess


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_mode(cfg: DictConfig) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "run") and hasattr(cfg.run, "optuna"):
            cfg.run.optuna.n_trials = 0
            cfg.run.optuna.enabled = False

        batch_size = max(1, int(cfg.run.training.batch_size))
        max_examples = max(2, batch_size * 2)
        dev_sel = max(1, min(int(cfg.run.dataset.split.dev_sel), max_examples // 2))
        dev_cal = max(1, min(int(cfg.run.dataset.split.dev_cal), max_examples - dev_sel))

        cfg.run.dataset.split.dev_sel = dev_sel
        cfg.run.dataset.split.dev_cal = dev_cal
        cfg.run.dataset.split.dev = dev_sel + dev_cal
        cfg.run.dataset.split.test = max(1, min(int(cfg.run.dataset.split.test), batch_size * 2))

        if not hasattr(cfg.run, "method_params") or cfg.run.method_params is None:
            cfg.run.method_params = OmegaConf.create({})
        mp = cfg.run.method_params
        if "dirichlet_samples" in mp:
            mp.dirichlet_samples = min(int(mp.dirichlet_samples), 200)
        if "per_mode_parsers" in mp:
            mp.per_mode_parsers = min(int(mp.per_mode_parsers), 2)
        mp.eval_parsers = min(int(mp.get("eval_parsers", 20)), 4)
        mp.eval_samples = min(int(mp.get("eval_samples", 20)), 4)
        mp.bootstrap_samples = min(int(mp.get("bootstrap_samples", 20)), 5)
        if "alpha0_unknown_mass" in mp:
            mp.unknown_mass_grid = [float(mp.alpha0_unknown_mass)]
        else:
            mp.unknown_mass_grid = [1.0]
        mp.audit_grid = mp.unknown_mass_grid
        cfg.run.training.epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


class WandbLogger:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.step = 0

    def log(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        if step is None:
            wandb.log(metrics, step=self.step)
            self.step += 1
        else:
            wandb.log(metrics, step=step)
            self.step = max(self.step, step + 1)


def label_list(kletter: str) -> List[str]:
    return [chr(i) for i in range(ord("A"), ord(kletter) + 1)]


def build_label_token_ids(tokenizer, kletter: str) -> Dict[str, int]:
    label_ids: Dict[str, int] = {}
    for label in label_list(kletter):
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Label '{label}' tokenizes into {len(tokens)} tokens; cannot train safely.")
        label_ids[label] = tokens[0]
    return label_ids


def build_supervised_dataloader(
    examples: List[Dict[str, Any]],
    tokenizer,
    prompt_template: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    label_token_ids: Dict[str, int],
):
    dataset = [(model_lib.build_prompt(prompt_template, ex), ex["gold"]) for ex in examples]

    def collate(batch):
        prompts = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()
        label_ids = torch.tensor([label_token_ids[l] for l in labels], dtype=torch.long)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label_ids = label_ids.to(device)
        return inputs, label_ids

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)


def train_supervised(
    model,
    tokenizer,
    train_examples: List[Dict[str, Any]],
    device: torch.device,
    cfg: DictConfig,
    logger: WandbLogger,
    is_encoder_decoder: bool,
    kletter: str,
) -> None:
    if cfg.run.training.inference_only or cfg.run.training.epochs <= 0:
        return

    prompt_template = model_lib.PROMPTS["direct_letter"]
    label_token_ids = build_label_token_ids(tokenizer, kletter)
    dataloader = build_supervised_dataloader(
        train_examples,
        tokenizer,
        prompt_template=prompt_template,
        batch_size=cfg.run.training.batch_size,
        max_length=cfg.run.dataset.max_length,
        device=device,
        label_token_ids=label_token_ids,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.run.training.learning_rate)

    def lr_lambda(step: int) -> float:
        warmup = max(1, int(cfg.run.training.warmup_steps))
        return min(1.0, step / warmup)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found in model.")

    decoder_start_token_id = None
    if is_encoder_decoder:
        decoder_start_token_id = model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id
            model.config.decoder_start_token_id = decoder_start_token_id
        assert decoder_start_token_id is not None, "decoder_start_token_id must be defined for encoder-decoder training."

    model.train()
    global_step = 0
    for epoch in range(int(cfg.run.training.epochs)):
        for step, (inputs, labels) in enumerate(dataloader):
            if step == 0:
                assert inputs["input_ids"].shape[0] == labels.shape[0]
                assert labels.ndim == 1
            optimizer.zero_grad(set_to_none=True)

            if is_encoder_decoder:
                batch_size = inputs["input_ids"].shape[0]
                decoder_input_ids = torch.full(
                    (batch_size, 1),
                    int(decoder_start_token_id),
                    device=device,
                    dtype=torch.long,
                )
                outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, 0, :]
            else:
                outputs = model(**inputs)
                logits = outputs.logits
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(inputs["input_ids"])
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(logits.size(0), device=logits.device)
                logits = logits[batch_idx, seq_lens]

            if step == 0:
                assert logits.shape[0] == labels.shape[0]
                assert logits.shape[-1] == model.config.vocab_size

            loss = F.cross_entropy(logits, labels)

            aux_grads = torch.autograd.grad(
                loss,
                params,
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )
            aux_norm = float(sum(g.abs().sum().item() for g in aux_grads if g is not None))

            loss.backward()
            grads = [p.grad for p in params]
            assert all(g is not None for g in grads), "Missing gradients before optimizer.step()"
            grad_norm = float(sum(g.abs().sum().item() for g in grads))
            assert grad_norm > 0.0, "Zero gradients detected before optimizer.step()"

            optimizer.step()
            scheduler.step()

            logger.log(
                {
                    "train/loss": float(loss.item()),
                    "train/grad_norm": grad_norm,
                    "train/aux_grad_norm": aux_norm,
                    "train/lr": float(scheduler.get_last_lr()[0]),
                    "train/epoch": epoch,
                    "train/step": global_step,
                }
            )
            global_step += 1

    model.eval()


def merge_method_params(cfg: DictConfig) -> Dict[str, Any]:
    defaults = cfg.method_defaults if hasattr(cfg, "method_defaults") else OmegaConf.create({})
    if not hasattr(cfg.run, "method_params") or cfg.run.method_params is None:
        cfg.run.method_params = OmegaConf.create({})
    merged = OmegaConf.merge(defaults, cfg.run.method_params)
    cfg.run.method_params = merged
    return OmegaConf.to_container(merged, resolve=True)


def resolve_method_type(method_name: str) -> str:
    name = method_name.lower()
    if "eb-cvar" in name or "empirical-bayes" in name:
        return "eb_cvar"
    if "brachs" in name:
        return "brachs"
    if "accuracy-only" in name:
        return "accuracy_only"
    if "chirps" in name:
        return "chirps"
    if "scohras" in name:
        return "scohras"
    raise ValueError(f"Unknown method name: {method_name}")


def run_optuna_search(
    cfg: DictConfig,
    method_params: Dict[str, Any],
    outputs_sel: Dict[str, List[str]],
    outputs_cal: Dict[str, List[str]],
    golds_sel: List[str],
    golds_cal: List[str],
    token_counts_cal: Dict[str, np.ndarray],
    kletter: str,
    history_modes: List[str],
    mode_names: List[str],
    pi_recent: Dict[str, float],
) -> Dict[str, float]:
    if not cfg.run.optuna.enabled or cfg.run.optuna.n_trials <= 0:
        return {}

    choices_alpha = None
    choices_quant = None
    for space in cfg.run.optuna.search_spaces:
        if space.param_name == "alpha0_unknown_mass":
            choices_alpha = space.choices
        if space.param_name == "posterior_quantile":
            choices_quant = space.choices
    if choices_alpha is None or choices_quant is None:
        raise ValueError("Optuna search spaces must include alpha0_unknown_mass and posterior_quantile")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        alpha0_unknown_mass = trial.suggest_categorical("alpha0_unknown_mass", choices_alpha)
        posterior_quantile = trial.suggest_categorical("posterior_quantile", choices_quant)
        alpha_post, _ = model_lib.compute_dirichlet_posterior(
            history_modes,
            mode_names,
            alpha0_unknown_mass=alpha0_unknown_mass,
            alpha0_base=method_params["alpha0_base"],
            include_unknown=method_params["unknown_bucket"],
        )
        selection = model_lib.select_brachs(
            outputs_sel=outputs_sel,
            outputs_cal=outputs_cal,
            golds_sel=golds_sel,
            golds_cal=golds_cal,
            token_counts_cal=token_counts_cal,
            kletter=kletter,
            alpha_post=alpha_post,
            mode_names=mode_names,
            invalid_eps=method_params["invalid_eps"],
            token_p95_budget=method_params["token_p95_budget"],
            alpha=method_params["alpha"],
            posterior_quantile=posterior_quantile,
            per_mode_parsers=method_params["per_mode_parsers"],
            dirichlet_samples=method_params["dirichlet_samples"],
            alpha0_unknown_mass=alpha0_unknown_mass,
            include_unknown=method_params["unknown_bucket"],
            seed=cfg.run.model.seed,
            conformal_tau=method_params["conformal_tau"],
        )
        chosen_prompt = selection["chosen_prompt"]
        eval_metrics = model_lib.drift_eval(
            outputs_cal[chosen_prompt],
            golds_cal,
            kletter,
            pi_recent,
            seed=cfg.run.model.seed,
            n_parsers=int(method_params["eval_parsers"]),
        )
        return float(eval_metrics["accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.run.optuna.n_trials, show_progress_bar=False)
    return study.best_params


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.run = OmegaConf.create(OmegaConf.to_container(cfg.run, resolve=True))
    configure_mode(cfg)

    run_cfg = cfg.run
    set_global_seed(run_cfg.model.seed)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", ".cache/")
    os.environ.setdefault("HF_DATASETS_CACHE", ".cache/")
    os.environ.setdefault("TRANSFORMERS_CACHE", ".cache/")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    wandb_enabled = cfg.wandb.mode != "disabled"
    if wandb_enabled:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        if wandb.run is not None:
            print(f"WandB URL: {wandb.run.get_url()}")

    logger = WandbLogger(enabled=wandb_enabled)

    method_params = merge_method_params(cfg)

    dev_sel, dev_cal, test, kletter = preprocess.load_mcqa_splits(run_cfg.dataset, run_cfg.model.seed)
    assert kletter in {"D", "E"}

    model, tokenizer, is_encoder_decoder = model_lib.load_model_and_tokenizer(
        run_cfg.model.name,
        device=device,
        cache_dir=".cache/",
    )
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set."
    assert model.config.vocab_size > 0, "Model vocab size must be positive."
    if not is_encoder_decoder:
        assert tokenizer.padding_side == "left"
    assert run_cfg.model.max_new_tokens > 0

    logger.log(
        {
            "init/model_name": run_cfg.model.name,
            "init/device": str(device),
            "init/is_encoder_decoder": float(is_encoder_decoder),
        }
    )

    if not run_cfg.training.inference_only and run_cfg.training.epochs > 0:
        train_supervised(
            model=model,
            tokenizer=tokenizer,
            train_examples=dev_sel,
            device=device,
            cfg=cfg,
            logger=logger,
            is_encoder_decoder=is_encoder_decoder,
            kletter=kletter,
        )

    golds_sel = [ex["gold"] for ex in dev_sel]
    golds_cal = [ex["gold"] for ex in dev_cal]
    golds_test = [ex["gold"] for ex in test]

    outputs_sel: Dict[str, List[str]] = {}
    outputs_cal: Dict[str, List[str]] = {}
    outputs_test: Dict[str, List[str]] = {}
    token_counts_cal: Dict[str, np.ndarray] = {}
    token_counts_test: Dict[str, np.ndarray] = {}

    def make_gen_logger(split: str, prompt: str):
        def _log(metrics: Dict[str, Any], step: int | None = None) -> None:
            logger.log({f"generation/{split}/{prompt}/" + k: v for k, v in metrics.items()}, step=step)

        return _log

    for prompt_name, prompt_template in model_lib.PROMPTS.items():
        outputs_sel[prompt_name], _ = model_lib.generate_outputs(
            model,
            tokenizer,
            dev_sel,
            prompt_template,
            max_new_tokens=run_cfg.model.max_new_tokens,
            batch_size=run_cfg.training.batch_size,
            device=device,
            max_length=run_cfg.dataset.max_length,
            is_encoder_decoder=is_encoder_decoder,
            log_fn=make_gen_logger("dev_sel", prompt_name),
        )
        outputs_cal[prompt_name], token_counts_cal[prompt_name] = model_lib.generate_outputs(
            model,
            tokenizer,
            dev_cal,
            prompt_template,
            max_new_tokens=run_cfg.model.max_new_tokens,
            batch_size=run_cfg.training.batch_size,
            device=device,
            max_length=run_cfg.dataset.max_length,
            is_encoder_decoder=is_encoder_decoder,
            log_fn=make_gen_logger("dev_cal", prompt_name),
        )
        outputs_test[prompt_name], token_counts_test[prompt_name] = model_lib.generate_outputs(
            model,
            tokenizer,
            test,
            prompt_template,
            max_new_tokens=run_cfg.model.max_new_tokens,
            batch_size=run_cfg.training.batch_size,
            device=device,
            max_length=run_cfg.dataset.max_length,
            is_encoder_decoder=is_encoder_decoder,
            log_fn=make_gen_logger("test", prompt_name),
        )

    mode_names = list(method_params.get("drift_modes", []))
    if not mode_names:
        raise ValueError("At least one drift mode must be specified in method_params.drift_modes")
    for mode in mode_names:
        if mode not in model_lib.MODE_PARSER_FACTORY:
            raise ValueError(f"Unknown drift mode in config: {mode}")

    drift_cfg = method_params.get("drift_distributions")
    if drift_cfg is None:
        raise ValueError("method_params.drift_distributions must be provided.")
    pi_old, pi_recent, pi_future = model_lib.prepare_drift_distributions(drift_cfg, mode_names)

    history_cfg = method_params.get("harness_history")
    if history_cfg is None:
        raise ValueError("method_params.harness_history must be provided.")
    history_modes = model_lib.simulate_harness_history(run_cfg.model.seed, mode_names, pi_old, pi_recent, history_cfg)

    include_unknown = bool(method_params.get("unknown_bucket", True))
    alpha0_base = float(method_params.get("alpha0_base", 0.5))

    use_empirical_bayes = bool(method_params.get("empirical_bayes_unknown_mass", False)) or (
        "empirical-bayes" in run_cfg.method.lower()
    )

    if use_empirical_bayes:
        grid = method_params.get("unknown_mass_grid", [0.5, 1, 2, 4, 8, 12])
        h_tr = int(history_cfg.get("train", max(1, len(history_modes) // 2)))
        alpha0_unknown_mass = model_lib.fit_unknown_mass(
            history_modes,
            mode_names,
            h_tr=h_tr,
            grid=grid,
        )
    else:
        alpha0_unknown_mass = float(method_params.get("alpha0_unknown_mass", 1.0))

    alpha_post, _ = model_lib.compute_dirichlet_posterior(
        history_modes,
        mode_names,
        alpha0_unknown_mass=alpha0_unknown_mass,
        alpha0_base=alpha0_base,
        include_unknown=include_unknown,
    )

    method_type = resolve_method_type(run_cfg.method)

    best_params: Dict[str, float] = {}
    if cfg.run.optuna.enabled and cfg.run.optuna.n_trials > 0 and cfg.mode == "full" and method_type == "brachs":
        best_params = run_optuna_search(
            cfg,
            method_params,
            outputs_sel,
            outputs_cal,
            golds_sel,
            golds_cal,
            token_counts_cal,
            kletter,
            history_modes,
            mode_names,
            pi_recent,
        )
        if best_params:
            alpha0_unknown_mass = float(best_params.get("alpha0_unknown_mass", alpha0_unknown_mass))
            method_params["alpha0_unknown_mass"] = alpha0_unknown_mass
            method_params["posterior_quantile"] = float(
                best_params.get("posterior_quantile", method_params.get("posterior_quantile", 0.1))
            )
            alpha_post, _ = model_lib.compute_dirichlet_posterior(
                history_modes,
                mode_names,
                alpha0_unknown_mass=alpha0_unknown_mass,
                alpha0_base=alpha0_base,
                include_unknown=include_unknown,
            )

    if method_type == "eb_cvar":
        selection = model_lib.select_eb_cvar_brachs(
            outputs_sel=outputs_sel,
            outputs_cal=outputs_cal,
            golds_sel=golds_sel,
            golds_cal=golds_cal,
            token_counts_cal=token_counts_cal,
            kletter=kletter,
            alpha_post=alpha_post,
            mode_names=mode_names,
            invalid_eps=method_params["invalid_eps"],
            token_p95_budget=method_params["token_p95_budget"],
            alpha=method_params["alpha"],
            cvar_delta=method_params["cvar_delta"],
            per_mode_parsers=method_params["per_mode_parsers"],
            dirichlet_samples=method_params["dirichlet_samples"],
            include_unknown=include_unknown,
            seed=run_cfg.model.seed,
            conformal_tau=method_params["conformal_tau"],
        )
    elif method_type == "brachs":
        selection = model_lib.select_brachs(
            outputs_sel=outputs_sel,
            outputs_cal=outputs_cal,
            golds_sel=golds_sel,
            golds_cal=golds_cal,
            token_counts_cal=token_counts_cal,
            kletter=kletter,
            alpha_post=alpha_post,
            mode_names=mode_names,
            invalid_eps=method_params["invalid_eps"],
            token_p95_budget=method_params["token_p95_budget"],
            alpha=method_params["alpha"],
            posterior_quantile=method_params.get("posterior_quantile", 0.1),
            per_mode_parsers=method_params["per_mode_parsers"],
            dirichlet_samples=method_params["dirichlet_samples"],
            alpha0_unknown_mass=alpha0_unknown_mass,
            include_unknown=include_unknown,
            seed=run_cfg.model.seed,
            conformal_tau=method_params["conformal_tau"],
        )
    elif method_type == "accuracy_only":
        selection = model_lib.select_accuracy_only(
            outputs_sel=outputs_sel,
            golds_sel=golds_sel,
            kletter=kletter,
            lenient_mode=method_params.get("lenient_mode", mode_names[0]),
            seed=run_cfg.model.seed,
        )
    elif method_type == "chirps":
        selection = model_lib.select_chirps(
            outputs_sel=outputs_sel,
            outputs_cal=outputs_cal,
            golds_sel=golds_sel,
            golds_cal=golds_cal,
            token_counts_cal=token_counts_cal,
            kletter=kletter,
            mode_names=mode_names,
            invalid_eps=method_params["invalid_eps"],
            token_p95_budget=method_params["token_p95_budget"],
            alpha=method_params["alpha"],
            per_mode_parsers=method_params["per_mode_parsers"],
            seed=run_cfg.model.seed,
            conformal_tau=method_params["conformal_tau"],
        )
    elif method_type == "scohras":
        selection = model_lib.select_scohras(
            outputs_sel=outputs_sel,
            outputs_cal=outputs_cal,
            golds_sel=golds_sel,
            golds_cal=golds_cal,
            token_counts_cal=token_counts_cal,
            kletter=kletter,
            mode_names=mode_names,
            invalid_eps=method_params["invalid_eps"],
            token_p95_budget=method_params["token_p95_budget"],
            alpha=method_params["alpha"],
            per_mode_parsers=method_params["per_mode_parsers"],
            pi_known=pi_recent,
            seed=run_cfg.model.seed,
            conformal_tau=method_params["conformal_tau"],
        )
    else:
        raise RuntimeError(f"Unhandled method_type: {method_type}")

    chosen_prompt = selection["chosen_prompt"]

    for prompt_name, metrics in selection["prompt_metrics"].items():
        logger.log({f"prompt/{prompt_name}/" + k: v for k, v in metrics.items()})

    eval_parsers = int(method_params.get("eval_parsers", 30))
    eval_future = model_lib.drift_eval(
        outputs_test[chosen_prompt],
        golds_test,
        kletter,
        pi_future,
        seed=run_cfg.model.seed,
        n_parsers=eval_parsers,
    )
    eval_recent = model_lib.drift_eval(
        outputs_test[chosen_prompt],
        golds_test,
        kletter,
        pi_recent,
        seed=run_cfg.model.seed + 1,
        n_parsers=eval_parsers,
    )

    drift_gap = eval_future["accuracy"] - eval_recent["accuracy"]
    token_p95_test = float(np.percentile(token_counts_test[chosen_prompt], 95))

    sample_count = int(method_params.get("eval_samples", 30))
    future_samples = model_lib.drift_eval_samples(
        outputs_test[chosen_prompt],
        golds_test,
        kletter,
        pi_future,
        seed=run_cfg.model.seed + 100,
        n_samples=sample_count,
    )
    recent_samples = model_lib.drift_eval_samples(
        outputs_test[chosen_prompt],
        golds_test,
        kletter,
        pi_recent,
        seed=run_cfg.model.seed + 110,
        n_samples=sample_count,
    )

    for acc, val in zip(future_samples["accuracy_samples"], future_samples["validity_samples"]):
        logger.log(
            {
                "eval/accuracy_pi_future_sample": float(acc),
                "eval/validity_pi_future_sample": float(val),
            }
        )

    for acc, val in zip(recent_samples["accuracy_samples"], recent_samples["validity_samples"]):
        logger.log(
            {
                "eval/accuracy_pi_recent_sample": float(acc),
                "eval/validity_pi_recent_sample": float(val),
            }
        )

    stability, prompt_probs = model_lib.selection_stability_bootstrap(
        method_type=method_type,
        outputs_sel=outputs_sel,
        outputs_cal=outputs_cal,
        golds_sel=golds_sel,
        golds_cal=golds_cal,
        token_counts_cal=token_counts_cal,
        kletter=kletter,
        alpha_post=alpha_post,
        mode_names=mode_names,
        invalid_eps=method_params["invalid_eps"],
        token_p95_budget=method_params["token_p95_budget"],
        alpha=method_params["alpha"],
        cvar_delta=method_params.get("cvar_delta", 0.05),
        posterior_quantile=method_params.get("posterior_quantile", 0.1),
        per_mode_parsers=method_params["per_mode_parsers"],
        dirichlet_samples=method_params["dirichlet_samples"],
        include_unknown=include_unknown,
        alpha0_unknown_mass=alpha0_unknown_mass,
        seed=run_cfg.model.seed,
        n_bootstrap=int(method_params.get("bootstrap_samples", 50)),
        conformal_tau=method_params["conformal_tau"],
        lenient_mode=method_params.get("lenient_mode", mode_names[0]),
        pi_known=pi_recent,
    )

    audit_grid = method_params.get("audit_grid", method_params.get("unknown_mass_grid", []))
    if include_unknown and audit_grid:
        audit_curve = model_lib.unknown_mass_audit_curve(
            outputs_sel=outputs_sel,
            outputs_cal=outputs_cal,
            outputs_test=outputs_test,
            golds_sel=golds_sel,
            golds_cal=golds_cal,
            golds_test=golds_test,
            token_counts_cal=token_counts_cal,
            kletter=kletter,
            mode_names=mode_names,
            history_modes=history_modes,
            invalid_eps=method_params["invalid_eps"],
            token_p95_budget=method_params["token_p95_budget"],
            alpha=method_params["alpha"],
            posterior_quantile=method_params.get("posterior_quantile", 0.1),
            per_mode_parsers=method_params["per_mode_parsers"],
            dirichlet_samples=method_params["dirichlet_samples"],
            grid=audit_grid,
            pi_future=pi_future,
            seed=run_cfg.model.seed,
            include_unknown=include_unknown,
            alpha0_base=alpha0_base,
            eval_parsers=eval_parsers,
            conformal_tau=method_params["conformal_tau"],
        )
    else:
        audit_curve = []

    labels_cm, cm = model_lib.confusion_matrix_from_outputs(
        outputs_test[chosen_prompt],
        golds_test,
        kletter,
        pi_future,
        seed=run_cfg.model.seed + 10,
        n_parsers=eval_parsers,
    )

    logger.log(
        {
            "eval/accuracy_pi_future": eval_future["accuracy"],
            "eval/validity_pi_future": eval_future["validity"],
            "eval/accuracy_pi_recent": eval_recent["accuracy"],
            "eval/validity_pi_recent": eval_recent["validity"],
            "eval/drift_generalization_gap": drift_gap,
            "eval/token_p95_test": token_p95_test,
            "eval/selection_stability_bootstrap": stability,
            "accuracy": eval_future["accuracy"],
        }
    )

    per_prompt_eval_metrics: Dict[str, Dict[str, float]] = {}
    per_prompt_token_p95: Dict[str, float] = {}
    for prompt_name in model_lib.PROMPTS:
        per_prompt_eval = model_lib.drift_eval(
            outputs_test[prompt_name],
            golds_test,
            kletter,
            pi_future,
            seed=run_cfg.model.seed + 5,
            n_parsers=eval_parsers,
        )
        per_prompt_eval_metrics[prompt_name] = per_prompt_eval
        per_prompt_token_p95[prompt_name] = float(np.percentile(token_counts_test[prompt_name], 95))
        logger.log(
            {
                f"prompt/{prompt_name}/accuracy_pi_future": per_prompt_eval["accuracy"],
                f"prompt/{prompt_name}/validity_pi_future": per_prompt_eval["validity"],
                f"prompt/{prompt_name}/token_p95_test": per_prompt_token_p95[prompt_name],
            }
        )

    if wandb_enabled:
        wandb.summary["eval/accuracy_pi_future"] = eval_future["accuracy"]
        wandb.summary["eval/validity_pi_future"] = eval_future["validity"]
        wandb.summary["eval/accuracy_pi_recent"] = eval_recent["accuracy"]
        wandb.summary["eval/validity_pi_recent"] = eval_recent["validity"]
        wandb.summary["eval/drift_generalization_gap"] = drift_gap
        wandb.summary["eval/token_p95_test"] = token_p95_test
        wandb.summary["eval/selection_stability_bootstrap"] = stability
        wandb.summary["eval/unknown_mass_audit_curve"] = audit_curve
        wandb.summary["eval/confusion_matrix"] = cm.tolist()
        wandb.summary["eval/confusion_matrix_labels"] = labels_cm
        wandb.summary["selection/chosen_prompt"] = chosen_prompt
        wandb.summary["selection/alpha0_unknown_mass"] = alpha0_unknown_mass
        wandb.summary["selection/prompt_probabilities"] = prompt_probs
        wandb.summary["accuracy"] = eval_future["accuracy"]
        wandb.summary["eval/accuracy_pi_future_samples"] = future_samples["accuracy_samples"]
        wandb.summary["eval/validity_pi_future_samples"] = future_samples["validity_samples"]
        wandb.summary["eval/accuracy_pi_recent_samples"] = recent_samples["accuracy_samples"]
        wandb.summary["eval/validity_pi_recent_samples"] = recent_samples["validity_samples"]
        if best_params:
            wandb.summary["optuna/best_params"] = best_params
        if "posterior_quantile" in method_params:
            wandb.summary["selection/posterior_quantile"] = method_params.get("posterior_quantile")

        for prompt_name in model_lib.PROMPTS:
            wandb.summary[f"prompt/{prompt_name}/accuracy_pi_future"] = per_prompt_eval_metrics[prompt_name][
                "accuracy"
            ]
            wandb.summary[f"prompt/{prompt_name}/validity_pi_future"] = per_prompt_eval_metrics[prompt_name][
                "validity"
            ]
            wandb.summary[f"prompt/{prompt_name}/token_p95_test"] = per_prompt_token_p95[prompt_name]

        if wandb.run is not None:
            print(f"WandB URL: {wandb.run.get_url()}")
        wandb.finish()
    else:
        print("WandB logging disabled (trial mode).")


if __name__ == "__main__":
    main()
