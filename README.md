# ccxp-decaptcha

A training pipeline for NTHU CCXP & CCXP OAuth decaptcha models.

|NTHU CCXP login captcha|NTHU CCXP OAuth captcha|
|---|---|
|<img width="186" height="320" alt="image" src="https://github.com/user-attachments/assets/677ae39c-89c4-49bd-8e4b-89eb49db3ac3" />|<img width="210" height="320" alt="image" src="https://github.com/user-attachments/assets/a31535ea-5ea4-4909-a8fd-4603f7e887b7" />|


Based on [nthu-ccxp-captcha](https://github.com/25349023/nthu-ccxp-captcha)'s training pipeline, this project:
- Rebuilt the original multi-window labeling workflow into a fully terminal-based labeling pipeline for faster manual data collection.
- Introduced a new six-head model architecture trained from scratch on a dataset of 30,000 images manually labeled in April 2026, achieving 99.96% six-digit sequence accuracy and 99.99% per-digit accuracy.
- Added a NTHU CCXP OAuth login decaptcha training pipeline with a 4-head attention-pooling model, achieving 98.79% four-digit sequence accuracy and 99.69% per-digit accuracy on a dataset of 23,000 Securimage PHP captcha images in May 2026.

## Setup

```bash
pip install -r requirements.txt
composer install # for Securimage PHP in the oauth pipeline
```

## Pipelines

### 1. Collect captcha data

CCXP:
```bash
python -m pipelines.ccxp.collect
```

OAuth:
```bash
python -m pipelines.oauth.collect
```

To relabel the last CCXP sample:
```bash
python -m pipelines.ccxp.relabel
```

### 2. Build dataset arrays

CCXP:
```bash
python -m pipelines.ccxp.build
```

OAuth:
```bash
python -m pipelines.oauth.build
```

### 3. Train and evaluate

CCXP:
```bash
python -m pipelines.ccxp.train
```

OAuth:
```bash
python -m pipelines.oauth.train
```

Model checkpoints are saved to `best.pt`.

## License

This project is released under the MIT License.

- Any unmodified code from [nthu-ccxp-captcha](https://github.com/25349023/nthu-ccxp-captcha) remains attributed to [25349023](https://github.com/25349023).
- Modifications and rewritten portions are additionally copyright (c) 2026 Hsi.
