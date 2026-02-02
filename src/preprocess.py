from typing import Dict, Any

from datasets import load_dataset

CACHE_DIR = ".cache/"


def format_openbookqa(example: Dict[str, Any]) -> Dict[str, Any]:
    question = example["question_stem"]
    choices = {c["label"]: c["text"] for c in example["choices"]}
    ordered_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join([f"{label}) {choices[label]}" for label in ordered_labels])
    return {
        "question": question,
        "choices": choices_text,
        "gold": example["answerKey"],
        "kletter": "D",
    }


def format_commonsenseqa(example: Dict[str, Any]) -> Dict[str, Any]:
    question = example["question"]
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    choices_text = "\n".join([f"{label}) {text}" for label, text in zip(labels, texts)])
    return {
        "question": question,
        "choices": choices_text,
        "gold": example["answerKey"],
        "kletter": "E",
    }


def load_mcqa_splits(dataset_cfg, seed: int):
    if dataset_cfg is None:
        raise ValueError("dataset configuration is required.")
    name = dataset_cfg.get("name") if isinstance(dataset_cfg, dict) else dataset_cfg.name
    if not name:
        raise ValueError("dataset.name is required in config.")
    name = name.lower()
    if name == "openbookqa":
        hf_name, hf_config, split_dev, split_test, formatter = (
            "openbookqa",
            "main",
            "validation",
            "test",
            format_openbookqa,
        )
    elif name in ("commonsenseqa", "commonsense_qa"):
        hf_name, hf_config, split_dev, split_test, formatter = (
            "tau/commonsense_qa",
            None,
            "validation",
            "test",
            format_commonsenseqa,
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    split_cfg = dataset_cfg.get("split") if isinstance(dataset_cfg, dict) else dataset_cfg.split
    if split_cfg is None:
        raise ValueError("dataset.split configuration is required.")

    dev_size = int(split_cfg["dev"])
    test_size = int(split_cfg["test"])
    dev_sel_size = int(split_cfg["dev_sel"])
    dev_cal_size = int(split_cfg["dev_cal"])

    if dev_sel_size + dev_cal_size > dev_size:
        raise ValueError("dev_sel + dev_cal must be <= dev size")

    dataset = load_dataset(hf_name, hf_config, cache_dir=CACHE_DIR) if hf_config else load_dataset(
        hf_name, cache_dir=CACHE_DIR
    )

    if dev_size > len(dataset[split_dev]):
        raise ValueError("Requested dev size exceeds available split length.")
    if test_size > len(dataset[split_test]):
        raise ValueError("Requested test size exceeds available split length.")

    dev_raw = dataset[split_dev].shuffle(seed=seed).select(range(dev_size))
    test_raw = dataset[split_test].shuffle(seed=seed + 1).select(range(test_size))

    dev_examples = [formatter(ex) for ex in dev_raw]
    test_examples = [formatter(ex) for ex in test_raw]

    if not dev_examples:
        raise ValueError("Dev examples are empty after preprocessing.")

    dev_sel = dev_examples[:dev_sel_size]
    dev_cal = dev_examples[dev_sel_size : dev_sel_size + dev_cal_size]

    kletter = dev_examples[0]["kletter"]
    return dev_sel, dev_cal, test_examples, kletter
