import json
import math
import re
import time
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import torch
from scipy.stats import beta
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

PROMPTS: Dict[str, str] = {
    "direct_letter": "{q}\nChoices:\n{choices}\nAnswer with a single letter.",
    "cot": "{q}\nChoices:\n{choices}\nLet's think step by step. Then answer with a single letter.",
    "cot_short": "{q}\nChoices:\n{choices}\nThink step by step in <=40 tokens. Then answer with a single letter.",
    "json_only": "{q}\nChoices:\n{choices}\nRespond ONLY as JSON: {{\"answer\":\"A\"}}.",
    "self_critique": "{q}\nChoices:\n{choices}\nReason, critique your reasoning, then give FINAL ANSWER as a single letter.",
    "json_then_letter": "{q}\nChoices:\n{choices}\nReturn JSON {{\"answer\":\"A\"}} then on a new line write FINAL=A.",
}


def load_model_and_tokenizer(model_name: str, device: torch.device, cache_dir: str = ".cache/"):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    added_tokens = 0
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            added_tokens = 1

    if is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    if tokenizer.pad_token_id is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None and getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id

    if added_tokens or len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()
    return model, tokenizer, is_encoder_decoder


def build_prompt(prompt_template: str, example: Dict[str, Any]) -> str:
    return prompt_template.format(q=example["question"], choices=example["choices"])


def generate_outputs(
    model,
    tokenizer,
    examples: List[Dict[str, Any]],
    prompt_template: str,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    max_length: int,
    is_encoder_decoder: bool,
    log_fn: Callable[[Dict[str, Any], int], None] | None = None,
) -> Tuple[List[str], np.ndarray]:
    prompts = [build_prompt(prompt_template, ex) for ex in examples]
    outputs: List[str] = []
    token_counts: List[int] = []

    start_time = time.time()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if start == 0:
            assert encoded["input_ids"].shape[0] == len(batch_prompts)
            assert encoded["input_ids"].shape == encoded["attention_mask"].shape
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        if is_encoder_decoder:
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            lengths = (generated != tokenizer.pad_token_id).sum(dim=1).cpu().numpy().tolist()
        else:
            input_len = encoded["input_ids"].shape[1]
            gen_tokens = generated[:, input_len:]
            decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            if tokenizer.pad_token_id is None:
                lengths = [int(seq.numel()) for seq in gen_tokens]
            else:
                lengths = (gen_tokens != tokenizer.pad_token_id).sum(dim=1).cpu().numpy().tolist()

        outputs.extend(decoded)
        token_counts.extend(lengths)

        if log_fn is not None:
            elapsed = time.time() - start_time
            log_fn(
                {
                    "batch_idx": start // batch_size,
                    "batch_size": len(batch_prompts),
                    "mean_new_tokens": float(np.mean(lengths)) if lengths else 0.0,
                    "elapsed_s": float(elapsed),
                },
                step=start // batch_size,
            )

    return outputs, np.asarray(token_counts, dtype=np.int32)


def letters_regex(kletter: str) -> str:
    return rf"\b([A-{kletter}])\b"


def parse_letter(text: str, kletter: str, extraction: str = "last", require_prefix: str = None, strip_punct: bool = False):
    t = text
    if require_prefix is not None:
        j = t.upper().find(require_prefix)
        if j < 0:
            return None, True
        t = t[j + len(require_prefix) :]
    t = t.upper()
    if strip_punct:
        t = re.sub(r"[^A-Z0-9\s{}\"\\:\,\[\]\-]", " ", t)
    letters = re.findall(letters_regex(kletter), t)
    if len(letters) == 0:
        return None, True
    if len(set(letters)) > 1:
        return None, True
    return (letters[-1] if extraction == "last" else letters[0]), False


def extract_first_json_obj(text: str):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        if text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json(
    text: str,
    kletter: str,
    strict_entire: bool = False,
    allow_trailing: bool = True,
    key: str = "answer",
    require_prefix: str = None,
):
    t = text.strip()
    if require_prefix is not None:
        j = t.upper().find(require_prefix)
        if j < 0:
            return None, True
        t = t[j + len(require_prefix) :].strip()
    if strict_entire:
        if not (t.startswith("{") and t.endswith("}")):
            return None, True
        blob = t
    else:
        blob = extract_first_json_obj(t) if allow_trailing else (t if t.startswith("{") and t.endswith("}") else None)
    if blob is None:
        return None, True
    try:
        obj = json.loads(blob)
    except Exception:
        return None, True
    ans = obj.get(key, None)
    if not isinstance(ans, str):
        return None, True
    ans = ans.strip().upper()
    if re.fullmatch(rf"[A-{kletter}]", ans) is None:
        return None, True
    return ans, False


MODE_PARSER_FACTORY: Dict[str, Callable] = {
    "M1_LETTER_LAST": lambda rng, kletter: lambda txt: parse_letter(
        txt, kletter, extraction="last", strip_punct=bool(rng.random() < 0.5)
    ),
    "M2_LETTER_FIRST": lambda rng, kletter: lambda txt: parse_letter(
        txt, kletter, extraction="first", strip_punct=bool(rng.random() < 0.5)
    ),
    "M3_LETTER_FINAL_PREFIX": lambda rng, kletter: lambda txt: parse_letter(
        txt, kletter, extraction="last", require_prefix="FINAL=", strip_punct=True
    ),
    "M4_JSON_EXTRACT_FIRST_BLOCK": lambda rng, kletter: lambda txt: parse_json(
        txt,
        kletter,
        strict_entire=False,
        allow_trailing=True,
        key="Answer" if rng.random() < 0.5 else "answer",
    ),
    "M5_JSON_STRICT_ENTIRE_OUTPUT": lambda rng, kletter: lambda txt: parse_json(
        txt, kletter, strict_entire=True, allow_trailing=False, key="answer"
    ),
    "M6_JSON_STRICT_ENTIRE_PLUS_PREFIX": lambda rng, kletter: lambda txt: parse_json(
        txt, kletter, strict_entire=True, allow_trailing=False, key="answer", require_prefix="FINAL="
    ),
}
MODE_PARSER_FACTORY["M4_JSON_EXTRACT_BLOCK"] = MODE_PARSER_FACTORY["M4_JSON_EXTRACT_FIRST_BLOCK"]


def sample_mode_parsers(rng: np.random.Generator, kletter: str, mode: str, n_parsers: int) -> List[Callable]:
    if mode not in MODE_PARSER_FACTORY:
        raise ValueError(f"Unknown drift mode: {mode}")
    return [MODE_PARSER_FACTORY[mode](rng, kletter) for _ in range(n_parsers)]


def pooled_counts(outputs: List[str], golds: List[str], parsers: List[Callable]) -> Tuple[int, int, int]:
    if len(outputs) != len(golds):
        raise ValueError("Outputs and golds length mismatch.")
    corr = 0
    inv = 0
    total = len(outputs) * len(parsers)
    for output, gold in zip(outputs, golds):
        for parser in parsers:
            pred, bad = parser(output)
            inv += int(bad)
            corr += int((not bad) and (pred == gold))
    return corr, inv, total


def clopper_pearson_ucb(k: int, n: int, alpha: float = 0.05) -> float:
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def wilson_lcb(k: int, n: int, alpha: float = 0.05) -> float:
    if n == 0:
        return 0.0
    z = 1.959963984540054
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    rad = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (center - rad) / denom)


def conformal_upper_quantile(x: np.ndarray, tau: float = 0.05) -> float:
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return 0.0
    k = int(math.ceil((n + 1) * (1 - tau)))
    k = min(max(k, 1), n)
    return float(np.sort(x)[k - 1])


def dirichlet_multinomial_logpred(counts_val: np.ndarray, alpha_post: np.ndarray) -> float:
    from math import lgamma

    a0 = float(np.sum(alpha_post))
    n = int(np.sum(counts_val))
    out = lgamma(a0) - lgamma(a0 + n)
    for c, a in zip(counts_val, alpha_post):
        out += lgamma(a + c) - lgamma(a)
    return out


def fit_unknown_mass(history_modes: List[str], mode_names: List[str], h_tr: int, grid: List[float]) -> float:
    if h_tr >= len(history_modes):
        raise ValueError("History train split must be smaller than total history size.")
    idx = {m: i for i, m in enumerate(mode_names)}
    hist_idx = [idx[m] if m in idx else len(mode_names) for m in history_modes]
    hist_idx = np.asarray(hist_idx, dtype=np.int32)

    tr = hist_idx[:h_tr]
    val = hist_idx[h_tr:]

    k_plus = len(mode_names) + 1
    c_tr = np.bincount(tr, minlength=k_plus).astype(np.float64)
    c_val = np.bincount(val, minlength=k_plus).astype(np.float64)

    best_mass, best_lp = None, -1e18
    for mass in grid:
        alpha0 = np.ones(k_plus, dtype=np.float64) * 0.5
        alpha0[-1] = mass
        alpha_post = alpha0 + c_tr
        lp = dirichlet_multinomial_logpred(c_val, alpha_post)
        if lp > best_lp:
            best_lp = lp
            best_mass = mass
    return float(best_mass)


def posterior_sample_pis(alpha_post: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha_post, size=n_samples)


def cvar(x: np.ndarray, tail: float = 0.05, upper: bool = False) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if upper:
        thr = np.quantile(x, 1 - tail)
        return float(x[x >= thr].mean())
    thr = np.quantile(x, tail)
    return float(x[x <= thr].mean())


def compute_dirichlet_posterior(
    history_modes: List[str],
    mode_names: List[str],
    alpha0_unknown_mass: float,
    alpha0_base: float = 0.5,
    include_unknown: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = {m: i for i, m in enumerate(mode_names)}
    k_plus = len(mode_names) + (1 if include_unknown else 0)
    counts = np.zeros(k_plus, dtype=np.float64)
    for mode in history_modes:
        if mode in idx:
            counts[idx[mode]] += 1
        elif include_unknown:
            counts[-1] += 1
    alpha0 = np.ones(k_plus, dtype=np.float64) * alpha0_base
    if include_unknown:
        alpha0[-1] = alpha0_unknown_mass
    return alpha0 + counts, counts


def normalize_distribution(pi: Dict[str, float], mode_names: List[str]) -> Dict[str, float]:
    filtered = {k: float(v) for k, v in pi.items() if k in mode_names}
    if not filtered:
        raise ValueError("Drift distribution has no overlap with configured mode names.")
    total = sum(filtered.values())
    return {k: v / total for k, v in filtered.items()}


def prepare_drift_distributions(drift_cfg: Dict[str, Any], mode_names: List[str]):
    if drift_cfg is None:
        raise ValueError("drift_distributions is required.")
    pi_old = normalize_distribution(drift_cfg.get("pi_old", {}), mode_names)
    pi_recent = normalize_distribution(drift_cfg.get("pi_recent", {}), mode_names)
    pi_future = normalize_distribution(drift_cfg.get("pi_future", {}), mode_names)
    return pi_old, pi_recent, pi_future


def simulate_harness_history(
    seed: int,
    mode_names: List[str],
    pi_old: Dict[str, float],
    pi_recent: Dict[str, float],
    harness_cfg: Dict[str, Any],
) -> List[str]:
    rng = np.random.default_rng(seed + 99)
    old_count = int(harness_cfg.get("old_count", harness_cfg.get("total", 0) // 2))
    recent_count = int(harness_cfg.get("recent_count", harness_cfg.get("total", 0) - old_count))
    total = int(harness_cfg.get("total", old_count + recent_count))
    if old_count + recent_count != total:
        recent_count = max(0, total - old_count)
    exclude = set(harness_cfg.get("exclude_modes", []))

    def sample_modes(pi: Dict[str, float], n: int) -> List[str]:
        filtered = {k: v for k, v in pi.items() if k in mode_names and k not in exclude}
        if not filtered:
            raise ValueError("No valid drift modes available after exclusions.")
        names = list(filtered.keys())
        probs = np.array([filtered[m] for m in names], dtype=np.float64)
        probs = probs / probs.sum()
        return rng.choice(names, size=n, p=probs).tolist()

    history = sample_modes(pi_old, old_count) + sample_modes(pi_recent, recent_count)
    if len(history) < total:
        history.extend(sample_modes(pi_recent, total - len(history)))
    return history[:total]


def select_eb_cvar_brachs(
    outputs_sel: Dict[str, List[str]],
    outputs_cal: Dict[str, List[str]],
    golds_sel: List[str],
    golds_cal: List[str],
    token_counts_cal: Dict[str, np.ndarray],
    kletter: str,
    alpha_post: np.ndarray,
    mode_names: List[str],
    invalid_eps: float,
    token_p95_budget: float,
    alpha: float,
    cvar_delta: float,
    per_mode_parsers: int,
    dirichlet_samples: int,
    include_unknown: bool,
    seed: int,
    conformal_tau: float,
) -> Dict[str, Any]:
    alpha_cal = alpha / max(1, len(PROMPTS))
    pis = posterior_sample_pis(alpha_post, dirichlet_samples, seed=seed + 11)
    rng_sel = np.random.default_rng(seed + 1)
    rng_cal = np.random.default_rng(seed + 2)

    best_prompt, best_obj = None, -1.0
    prompt_metrics: Dict[str, Dict[str, float]] = {}

    for prompt in PROMPTS:
        p95 = conformal_upper_quantile(token_counts_cal[prompt], tau=conformal_tau)
        if p95 > token_p95_budget:
            prompt_metrics[prompt] = {"token_p95_cal": p95, "constraint_pass": 0.0}
            continue

        inv_ucb = []
        acc_lcb = []
        for mode in mode_names:
            parsers_cal = sample_mode_parsers(rng_cal, kletter, mode, per_mode_parsers)
            corr_c, inv_c, tot_c = pooled_counts(outputs_cal[prompt], golds_cal, parsers_cal)
            inv_ucb.append(clopper_pearson_ucb(inv_c, tot_c, alpha=alpha_cal))

            parsers_sel = sample_mode_parsers(rng_sel, kletter, mode, per_mode_parsers)
            corr_s, inv_s, tot_s = pooled_counts(outputs_sel[prompt], golds_sel, parsers_sel)
            acc_lcb.append(wilson_lcb(corr_s, tot_s, alpha=alpha))

        if include_unknown:
            inv_ucb.append(1.0)
            acc_lcb.append(0.0)

        inv_ucb = np.asarray(inv_ucb)
        acc_lcb = np.asarray(acc_lcb)

        inv_risk = pis @ inv_ucb
        acc_risk = pis @ acc_lcb

        inv_cvar = cvar(inv_risk, tail=cvar_delta, upper=True)
        acc_cvar = cvar(acc_risk, tail=cvar_delta, upper=False)

        constraint_pass = float(inv_cvar <= invalid_eps)
        prompt_metrics[prompt] = {
            "token_p95_cal": p95,
            "inv_cvar": inv_cvar,
            "acc_cvar": acc_cvar,
            "constraint_pass": constraint_pass,
        }

        if constraint_pass and acc_cvar > best_obj:
            best_obj = acc_cvar
            best_prompt = prompt

    if best_prompt is None:
        best_prompt = max(prompt_metrics.keys(), key=lambda p: prompt_metrics[p].get("acc_cvar", -1.0))

    return {"chosen_prompt": best_prompt, "prompt_metrics": prompt_metrics}


def select_brachs(
    outputs_sel: Dict[str, List[str]],
    outputs_cal: Dict[str, List[str]],
    golds_sel: List[str],
    golds_cal: List[str],
    token_counts_cal: Dict[str, np.ndarray],
    kletter: str,
    alpha_post: np.ndarray,
    mode_names: List[str],
    invalid_eps: float,
    token_p95_budget: float,
    alpha: float,
    posterior_quantile: float,
    per_mode_parsers: int,
    dirichlet_samples: int,
    alpha0_unknown_mass: float,
    include_unknown: bool,
    seed: int,
    conformal_tau: float,
) -> Dict[str, Any]:
    alpha_cal = alpha / max(1, len(PROMPTS))
    pis = posterior_sample_pis(alpha_post, dirichlet_samples, seed=seed + 13)
    rng_sel = np.random.default_rng(seed + 3)
    rng_cal = np.random.default_rng(seed + 4)

    best_prompt, best_obj = None, -1.0
    prompt_metrics: Dict[str, Dict[str, float]] = {}

    for prompt in PROMPTS:
        p95 = conformal_upper_quantile(token_counts_cal[prompt], tau=conformal_tau)
        if p95 > token_p95_budget:
            prompt_metrics[prompt] = {"token_p95_cal": p95, "constraint_pass": 0.0}
            continue

        inv_ucb = []
        acc_lcb = []
        for mode in mode_names:
            parsers_cal = sample_mode_parsers(rng_cal, kletter, mode, per_mode_parsers)
            corr_c, inv_c, tot_c = pooled_counts(outputs_cal[prompt], golds_cal, parsers_cal)
            inv_ucb.append(clopper_pearson_ucb(inv_c, tot_c, alpha=alpha_cal))

            parsers_sel = sample_mode_parsers(rng_sel, kletter, mode, per_mode_parsers)
            corr_s, inv_s, tot_s = pooled_counts(outputs_sel[prompt], golds_sel, parsers_sel)
            acc_lcb.append(wilson_lcb(corr_s, tot_s, alpha=alpha))

        if include_unknown:
            inv_ucb.append(1.0)
            acc_lcb.append(0.0)

        inv_ucb = np.asarray(inv_ucb)
        acc_lcb = np.asarray(acc_lcb)

        inv_risk = pis @ inv_ucb
        acc_risk = pis @ acc_lcb

        inv_q = float(np.quantile(inv_risk, 1 - posterior_quantile))
        acc_q = float(np.quantile(acc_risk, posterior_quantile))

        constraint_pass = float(inv_q <= invalid_eps)
        prompt_metrics[prompt] = {
            "token_p95_cal": p95,
            "inv_quantile": inv_q,
            "acc_quantile": acc_q,
            "constraint_pass": constraint_pass,
        }

        if constraint_pass and acc_q > best_obj:
            best_obj = acc_q
            best_prompt = prompt

    if best_prompt is None:
        best_prompt = max(prompt_metrics.keys(), key=lambda p: prompt_metrics[p].get("acc_quantile", -1.0))

    return {"chosen_prompt": best_prompt, "prompt_metrics": prompt_metrics}


def select_accuracy_only(
    outputs_sel: Dict[str, List[str]],
    golds_sel: List[str],
    kletter: str,
    lenient_mode: str,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 77)
    parser = MODE_PARSER_FACTORY[lenient_mode](rng, kletter)
    prompt_metrics: Dict[str, Dict[str, float]] = {}
    for prompt in PROMPTS:
        correct = 0
        valid = 0
        for output, gold in zip(outputs_sel[prompt], golds_sel):
            pred, bad = parser(output)
            if not bad:
                valid += 1
                correct += int(pred == gold)
        accuracy_valid = correct / valid if valid > 0 else 0.0
        valid_rate = valid / len(outputs_sel[prompt]) if outputs_sel[prompt] else 0.0
        prompt_metrics[prompt] = {"accuracy_valid_only": accuracy_valid, "valid_rate": valid_rate}

    chosen_prompt = max(prompt_metrics.keys(), key=lambda p: prompt_metrics[p]["accuracy_valid_only"])
    return {"chosen_prompt": chosen_prompt, "prompt_metrics": prompt_metrics}


def select_chirps(
    outputs_sel: Dict[str, List[str]],
    outputs_cal: Dict[str, List[str]],
    golds_sel: List[str],
    golds_cal: List[str],
    token_counts_cal: Dict[str, np.ndarray],
    kletter: str,
    mode_names: List[str],
    invalid_eps: float,
    token_p95_budget: float,
    alpha: float,
    per_mode_parsers: int,
    seed: int,
    conformal_tau: float,
) -> Dict[str, Any]:
    alpha_cal = alpha / max(1, len(PROMPTS))
    rng_sel = np.random.default_rng(seed + 21)
    rng_cal = np.random.default_rng(seed + 22)

    best_prompt, best_obj = None, -1.0
    prompt_metrics: Dict[str, Dict[str, float]] = {}

    for prompt in PROMPTS:
        p95 = conformal_upper_quantile(token_counts_cal[prompt], tau=conformal_tau)
        if p95 > token_p95_budget:
            prompt_metrics[prompt] = {"token_p95_cal": p95, "constraint_pass": 0.0}
            continue

        parsers_cal = []
        parsers_sel = []
        for mode in mode_names:
            parsers_cal.extend(sample_mode_parsers(rng_cal, kletter, mode, per_mode_parsers))
            parsers_sel.extend(sample_mode_parsers(rng_sel, kletter, mode, per_mode_parsers))

        corr_c, inv_c, tot_c = pooled_counts(outputs_cal[prompt], golds_cal, parsers_cal)
        inv_ucb = clopper_pearson_ucb(inv_c, tot_c, alpha=alpha_cal)
        corr_s, inv_s, tot_s = pooled_counts(outputs_sel[prompt], golds_sel, parsers_sel)
        acc_lcb = wilson_lcb(corr_s, tot_s, alpha=alpha)

        constraint_pass = float(inv_ucb <= invalid_eps)
        prompt_metrics[prompt] = {
            "token_p95_cal": p95,
            "inv_ucb": inv_ucb,
            "acc_lcb": acc_lcb,
            "constraint_pass": constraint_pass,
        }

        if constraint_pass and acc_lcb > best_obj:
            best_obj = acc_lcb
            best_prompt = prompt

    if best_prompt is None:
        best_prompt = max(prompt_metrics.keys(), key=lambda p: prompt_metrics[p].get("acc_lcb", -1.0))

    return {"chosen_prompt": best_prompt, "prompt_metrics": prompt_metrics}


def select_scohras(
    outputs_sel: Dict[str, List[str]],
    outputs_cal: Dict[str, List[str]],
    golds_sel: List[str],
    golds_cal: List[str],
    token_counts_cal: Dict[str, np.ndarray],
    kletter: str,
    mode_names: List[str],
    invalid_eps: float,
    token_p95_budget: float,
    alpha: float,
    per_mode_parsers: int,
    pi_known: Dict[str, float],
    seed: int,
    conformal_tau: float,
) -> Dict[str, Any]:
    alpha_cal = alpha / max(1, len(PROMPTS))
    rng_sel = np.random.default_rng(seed + 31)
    rng_cal = np.random.default_rng(seed + 32)

    weights = np.array([pi_known[m] for m in mode_names], dtype=np.float64)
    weights = weights / weights.sum()

    best_prompt, best_obj = None, -1.0
    prompt_metrics: Dict[str, Dict[str, float]] = {}

    for prompt in PROMPTS:
        p95 = conformal_upper_quantile(token_counts_cal[prompt], tau=conformal_tau)
        if p95 > token_p95_budget:
            prompt_metrics[prompt] = {"token_p95_cal": p95, "constraint_pass": 0.0}
            continue

        inv_ucb = []
        acc_lcb = []
        for mode in mode_names:
            parsers_cal = sample_mode_parsers(rng_cal, kletter, mode, per_mode_parsers)
            corr_c, inv_c, tot_c = pooled_counts(outputs_cal[prompt], golds_cal, parsers_cal)
            inv_ucb.append(clopper_pearson_ucb(inv_c, tot_c, alpha=alpha_cal))

            parsers_sel = sample_mode_parsers(rng_sel, kletter, mode, per_mode_parsers)
            corr_s, inv_s, tot_s = pooled_counts(outputs_sel[prompt], golds_sel, parsers_sel)
            acc_lcb.append(wilson_lcb(corr_s, tot_s, alpha=alpha))

        inv_ucb = np.asarray(inv_ucb)
        acc_lcb = np.asarray(acc_lcb)
        inv_risk = float(weights @ inv_ucb)
        acc_risk = float(weights @ acc_lcb)

        constraint_pass = float(inv_risk <= invalid_eps)
        prompt_metrics[prompt] = {
            "token_p95_cal": p95,
            "inv_risk": inv_risk,
            "acc_risk": acc_risk,
            "constraint_pass": constraint_pass,
        }

        if constraint_pass and acc_risk > best_obj:
            best_obj = acc_risk
            best_prompt = prompt

    if best_prompt is None:
        best_prompt = max(prompt_metrics.keys(), key=lambda p: prompt_metrics[p].get("acc_risk", -1.0))

    return {"chosen_prompt": best_prompt, "prompt_metrics": prompt_metrics}


def drift_eval(outputs: List[str], golds: List[str], kletter: str, pi: Dict[str, float], seed: int, n_parsers: int):
    rng = np.random.default_rng(seed)
    mode_names = list(pi.keys())
    probs = np.array([pi[m] for m in mode_names], dtype=np.float64)
    probs = probs / probs.sum()

    parsers = []
    for _ in range(n_parsers):
        mode = rng.choice(mode_names, p=probs)
        parsers.append(MODE_PARSER_FACTORY[mode](rng, kletter))

    corr, inv, tot = pooled_counts(outputs, golds, parsers)
    return {"accuracy": corr / tot, "validity": 1.0 - inv / tot}


def drift_eval_samples(
    outputs: List[str],
    golds: List[str],
    kletter: str,
    pi: Dict[str, float],
    seed: int,
    n_samples: int,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    mode_names = list(pi.keys())
    probs = np.array([pi[m] for m in mode_names], dtype=np.float64)
    probs = probs / probs.sum()
    accuracy_samples: List[float] = []
    validity_samples: List[float] = []

    for _ in range(n_samples):
        mode = rng.choice(mode_names, p=probs)
        parser = MODE_PARSER_FACTORY[mode](rng, kletter)
        corr = 0
        inv = 0
        for output, gold in zip(outputs, golds):
            pred, bad = parser(output)
            inv += int(bad)
            corr += int((not bad) and (pred == gold))
        total = len(outputs)
        accuracy_samples.append(corr / total if total > 0 else 0.0)
        validity_samples.append(1.0 - inv / total if total > 0 else 0.0)

    return {"accuracy_samples": accuracy_samples, "validity_samples": validity_samples}


def selection_stability_bootstrap(
    method_type: str,
    outputs_sel: Dict[str, List[str]],
    outputs_cal: Dict[str, List[str]],
    golds_sel: List[str],
    golds_cal: List[str],
    token_counts_cal: Dict[str, np.ndarray],
    kletter: str,
    alpha_post: np.ndarray,
    mode_names: List[str],
    invalid_eps: float,
    token_p95_budget: float,
    alpha: float,
    cvar_delta: float,
    posterior_quantile: float,
    per_mode_parsers: int,
    dirichlet_samples: int,
    include_unknown: bool,
    alpha0_unknown_mass: float,
    seed: int,
    n_bootstrap: int,
    conformal_tau: float,
    lenient_mode: str,
    pi_known: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    rng = np.random.default_rng(seed + 50)
    counts = {prompt: 0 for prompt in PROMPTS}

    for b in range(n_bootstrap):
        idx = rng.integers(0, len(golds_sel), size=len(golds_sel))
        resampled_outputs = {prompt: [outputs_sel[prompt][i] for i in idx] for prompt in PROMPTS}
        resampled_golds = [golds_sel[i] for i in idx]

        if method_type == "eb_cvar":
            selection = select_eb_cvar_brachs(
                outputs_sel=resampled_outputs,
                outputs_cal=outputs_cal,
                golds_sel=resampled_golds,
                golds_cal=golds_cal,
                token_counts_cal=token_counts_cal,
                kletter=kletter,
                alpha_post=alpha_post,
                mode_names=mode_names,
                invalid_eps=invalid_eps,
                token_p95_budget=token_p95_budget,
                alpha=alpha,
                cvar_delta=cvar_delta,
                per_mode_parsers=per_mode_parsers,
                dirichlet_samples=dirichlet_samples,
                include_unknown=include_unknown,
                seed=seed + b,
                conformal_tau=conformal_tau,
            )
        elif method_type == "brachs":
            selection = select_brachs(
                outputs_sel=resampled_outputs,
                outputs_cal=outputs_cal,
                golds_sel=resampled_golds,
                golds_cal=golds_cal,
                token_counts_cal=token_counts_cal,
                kletter=kletter,
                alpha_post=alpha_post,
                mode_names=mode_names,
                invalid_eps=invalid_eps,
                token_p95_budget=token_p95_budget,
                alpha=alpha,
                posterior_quantile=posterior_quantile,
                per_mode_parsers=per_mode_parsers,
                dirichlet_samples=dirichlet_samples,
                alpha0_unknown_mass=alpha0_unknown_mass,
                include_unknown=include_unknown,
                seed=seed + b,
                conformal_tau=conformal_tau,
            )
        elif method_type == "accuracy_only":
            selection = select_accuracy_only(
                outputs_sel=resampled_outputs,
                golds_sel=resampled_golds,
                kletter=kletter,
                lenient_mode=lenient_mode,
                seed=seed + b,
            )
        elif method_type == "chirps":
            selection = select_chirps(
                outputs_sel=resampled_outputs,
                outputs_cal=outputs_cal,
                golds_sel=resampled_golds,
                golds_cal=golds_cal,
                token_counts_cal=token_counts_cal,
                kletter=kletter,
                mode_names=mode_names,
                invalid_eps=invalid_eps,
                token_p95_budget=token_p95_budget,
                alpha=alpha,
                per_mode_parsers=per_mode_parsers,
                seed=seed + b,
                conformal_tau=conformal_tau,
            )
        elif method_type == "scohras":
            selection = select_scohras(
                outputs_sel=resampled_outputs,
                outputs_cal=outputs_cal,
                golds_sel=resampled_golds,
                golds_cal=golds_cal,
                token_counts_cal=token_counts_cal,
                kletter=kletter,
                mode_names=mode_names,
                invalid_eps=invalid_eps,
                token_p95_budget=token_p95_budget,
                alpha=alpha,
                per_mode_parsers=per_mode_parsers,
                pi_known=pi_known,
                seed=seed + b,
                conformal_tau=conformal_tau,
            )
        else:
            raise ValueError(f"Unknown method_type: {method_type}")

        counts[selection["chosen_prompt"]] += 1

    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()} if total > 0 else {}
    stability = max(probs.values()) if probs else 0.0
    return stability, probs


def unknown_mass_audit_curve(
    outputs_sel: Dict[str, List[str]],
    outputs_cal: Dict[str, List[str]],
    outputs_test: Dict[str, List[str]],
    golds_sel: List[str],
    golds_cal: List[str],
    golds_test: List[str],
    token_counts_cal: Dict[str, np.ndarray],
    kletter: str,
    mode_names: List[str],
    history_modes: List[str],
    invalid_eps: float,
    token_p95_budget: float,
    alpha: float,
    posterior_quantile: float,
    per_mode_parsers: int,
    dirichlet_samples: int,
    grid: List[float],
    pi_future: Dict[str, float],
    seed: int,
    include_unknown: bool,
    alpha0_base: float,
    eval_parsers: int,
    conformal_tau: float,
) -> List[Dict[str, float]]:
    curve = []
    for mass in grid:
        alpha_post, _ = compute_dirichlet_posterior(
            history_modes,
            mode_names,
            alpha0_unknown_mass=mass,
            alpha0_base=alpha0_base,
            include_unknown=include_unknown,
        )
        selection = select_brachs(
            outputs_sel=outputs_sel,
            outputs_cal=outputs_cal,
            golds_sel=golds_sel,
            golds_cal=golds_cal,
            token_counts_cal=token_counts_cal,
            kletter=kletter,
            alpha_post=alpha_post,
            mode_names=mode_names,
            invalid_eps=invalid_eps,
            token_p95_budget=token_p95_budget,
            alpha=alpha,
            posterior_quantile=posterior_quantile,
            per_mode_parsers=per_mode_parsers,
            dirichlet_samples=dirichlet_samples,
            alpha0_unknown_mass=mass,
            include_unknown=include_unknown,
            seed=seed,
            conformal_tau=conformal_tau,
        )
        chosen_prompt = selection["chosen_prompt"]
        eval_metrics = drift_eval(
            outputs_test[chosen_prompt],
            golds_test,
            kletter,
            pi_future,
            seed=seed + 10,
            n_parsers=eval_parsers,
        )
        curve.append({"alpha0_unknown_mass": mass, "accuracy": eval_metrics["accuracy"]})
    return curve


def list_labels(kletter: str) -> List[str]:
    return [chr(i) for i in range(ord("A"), ord(kletter) + 1)] + ["INVALID"]


def confusion_matrix_from_outputs(
    outputs: List[str],
    golds: List[str],
    kletter: str,
    pi: Dict[str, float],
    seed: int,
    n_parsers: int,
) -> Tuple[List[str], np.ndarray]:
    labels = list_labels(kletter)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    rng = np.random.default_rng(seed)
    mode_names = list(pi.keys())
    probs = np.array([pi[m] for m in mode_names], dtype=np.float64)
    probs = probs / probs.sum()

    for _ in range(n_parsers):
        mode = rng.choice(mode_names, p=probs)
        parser = MODE_PARSER_FACTORY[mode](rng, kletter)
        for output, gold in zip(outputs, golds):
            pred, bad = parser(output)
            gold_label = gold if gold in label_to_idx else "INVALID"
            pred_label = "INVALID" if bad else pred
            matrix[label_to_idx[gold_label], label_to_idx[pred_label]] += 1

    return labels, matrix
