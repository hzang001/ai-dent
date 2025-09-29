# AI for Dent – AI Modeling Prototype

This repository now includes a first-pass machine learning workflow that trains an oral-cancer screening model against a public biomarker-style dataset.

## Dataset
- Source: [`ahmedshaban26/oral-cancer`](https://github.com/ahmedshaban26/oral-cancer) (CSV of 716 saliva/metabolite measurements labeled for oral cancer).
- Features: eight continuous biomarker intensities per participant; 24 values were missing in `biomarker_4` and `biomarker_5` and are imputed with the column median during preprocessing.
- Label: `1` denotes confirmed oral cancer, `0` indicates control/high-risk without cancer.

## Training Pipeline
Run the end-to-end experiment from the project root:

```bash
python src/train_oral_cancer_model.py \
  --data-path data/oral_cancer.csv \
  --output-dir artifacts
```

What the script does:
- Stratified 80/20 train/hold-out split with fixed seed (`42`).
- Nested cross-validation (5 outer × 3 inner folds) to tune a class-weighted logistic regression with L1/L2 penalty search.
- Median imputation + standardisation are included in the pipeline for reproducibility.
- Retrains the best configuration on the full training split and reports metrics on the untouched hold-out set.
- Saves artefacts into `artifacts/`: serialized model (`models/log_reg_pipeline.joblib`), cross-validation report (`reports/model_performance.json`), permutation importances, and coefficient tables.

## Current Results (hold-out set)
- ROC AUC: **0.67**
- Average precision: **0.36**
- Sensitivity: **0.64**
- Specificity: **0.64**

Refer to `artifacts/reports/model_performance.json` for full fold-level statistics and to `artifacts/reports/feature_coefficients.csv` for feature effects.

## Suggested Next Steps
1. Re-introduce tree-based ensembles (HistGradientBoosting/RandomForest) once parallel backends are allowed or replace with single-threaded implementations to compare performance.
2. Engineer domain-driven features (cytokine ratios, microbiome diversity metrics) when richer datasets become available.
3. Add decision-threshold optimisation and calibration to improve clinical interpretability (e.g., maximise sensitivity at ≥80% specificity).
4. Wrap the script in automated tests/CI and expose a lightweight prediction API for downstream integration into screening workflows.


## LLM Fine-Tuning Workflow
Build an instruction-style dataset from the same biomarkers and fine-tune a lightweight chat LLM with LoRA adapters for risk triage narratives.

### 1. Install dependencies
Use a Python environment with CUDA-enabled PyTorch if you intend to train on GPU.

```
pip install --upgrade "transformers>=4.40" "datasets>=2.18" "peft>=0.11" accelerate sentencepiece
# Optional for 8-bit / 4-bit quantisation
pip install bitsandbytes
```

### 2. Generate prompt/response data

```
python src/prepare_llm_data.py \
  --data-path data/oral_cancer.csv \
  --output-dir data/llm
```

This writes `train.jsonl`, `valid.jsonl`, and `test.jsonl` where each record contains a biomarker prompt and a short answer (`Cancer` or `Control`).

### 3. Run LoRA fine-tuning

```
python src/fine_tune_llm.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-file data/llm/train.jsonl \
  --valid-file data/llm/valid.jsonl \
  --output-dir artifacts/llm \
  --load-in-4bit  # optional, requires bitsandbytes
```

Key artefacts:
- `artifacts/llm/lora_adapter/`: LoRA weights ready to merge with the base checkpoint.
- `artifacts/llm/tokenizer/`: tokenizer snapshot aligned with the adapter.
- `artifacts/llm/validation_metrics.json`: simple accuracy from greedy generation on the validation prompts.

### 4. Use the fine-tuned adapter

```python
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = Path("artifacts/llm/lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("artifacts/llm/tokenizer")
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)

prompt = """### Instruction
Determine whether the saliva biomarker profile indicates oral cancer.
### Biomarkers
biomarker_1: 5.12
biomarker_2: 12.45
biomarker_3: 0.98
biomarker_4: 23.75
biomarker_5: 118.2
biomarker_6: 1.76
biomarker_7: 450.0
biomarker_8: 0.99
### Response"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=4, temperature=0.0)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Hardware considerations
- Expect the 1.1B TinyLlama adapter to fit comfortably on a single 16 GB GPU when 4-bit quantisation is enabled.
- For larger models (e.g., Llama-2/3 7B chat), adjust batch size, gradient accumulation, and enable `--load-in-4bit`.
- CPU-only fine-tuning is not recommended; inference with the trained adapter is still feasible on CPU once weights are saved.
