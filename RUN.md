# How to Run

## 1. Setup (first time only)

```bash
cd "/Users/pranavks/Quantitative Trading"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Run

```bash
cd "/Users/pranavks/Quantitative Trading"
source venv/bin/activate
python main.py
```

## 3. Options

| Flag | Description |
|------|-------------|
| `--provider csv` | Use CSV file instead of synthetic data |
| `--provider kite` | Use Kite live data |
| `--cpcv` | Run CPCV cross-validation |
| `--report` | Save trade log to `trade_log.csv` |
| `--rl` | Use RL executor (if model exists) |
| `--train-rl` | Train RL agent first, then backtest |

## 4. Examples

```bash
python main.py --provider csv
python main.py --cpcv --report
python main.py --train-rl
```
