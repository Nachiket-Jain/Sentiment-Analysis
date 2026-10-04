# Spelling-variant consistency as an uncertainty signal for Hinglish sentiment

Files
- `prepare_sentimix.py`     CoNLL -> CSV, adds code-mixing index (CMI) per tweet
- `variants.py`             rule-based romanization-variant generator (Hin-tagged words only)
- `build_variant_files.py`  pre-generates fixed variants (train k=4, dev/test k=8)
- `train_predict.py`        fine-tune + dump logits for originals and variants (GPU)
- `analyze_uncertainty.py`  error-detection comparison: MSP vs variant-disagreement vs ensemble

## Local (CPU) steps, already done
    python reliability/prepare_sentimix.py
    python reliability/build_variant_files.py

## Kaggle / Colab (GPU) steps
1. Upload as a Kaggle Dataset (e.g. `sentimix-proc`): everything in `data/sentimix/processed/`
   (train/dev/test.csv + *_variants.json) and `reliability/train_predict.py`.
2. Smoke test (1 min):
       python train_predict.py --model xlm-roberta-base --data /kaggle/input/sentimix-proc \
              --out /kaggle/working/out --limit 200 --epochs 1
3. Real runs (~15-25 min each on a T4). 3 seeds x 2 models, plus the augmentation baseline:
       for s in 0 1 2; do
         python train_predict.py --model xlm-roberta-base  --seed $s --data ... --out ...
         python train_predict.py --model ai4bharat/IndicBERTv2-MLM-only  (original indic-bert is now gated; MuRIL: google/muril-base-cased) --seed $s --data ... --out ...
       done
       python train_predict.py --model xlm-roberta-base --seed 0 --augment --data ... --out ...
4. Download `out/` and put it in `reliability/out/`, then:
       python reliability/analyze_uncertainty.py reliability/out xlm-roberta-base_s0 xlm-roberta-base_s1 xlm-roberta-base_s2
