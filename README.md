# Indoor Positioning Final Submission
12223648 장수민 

## Files

| File | Role |
|---|---|
| `main.py` | Final inference code executed by the grader |
| `train.py` | Training script that creates `model.pkl` |
| `model.pkl` | Saved model package used by `main.py` |
| `report.md` | Final report |
| `README.md` | File description and execution guide |

## Standard Environment

This submission uses only the packages allowed in the standard grading environment.

| Package | Usage |
|---|---|
| `numpy` | array computation |
| `scipy.io` | `.mat` file loading |
| `scikit-learn` | Random Forest models and preprocessing |

No additional external package is required, so `requirements.txt` is not needed.

## How to Train

Place `train.py` and `DH_FR1.mat` in the same directory, then run:

```bash
python train.py
```

This creates:

```text
model.pkl
```

`train.py` uses the provided ground-truth position `p` only during training.

## How to Run Inference

Place the following files in the same directory:

```text
main.py
model.pkl
DH_FR1.mat
```

Then run:

```bash
python main.py
```

The script may finish without printing anything. That is normal.  
The grader calls `main()` and checks the returned NumPy array.

## main.py Specification

The required function signature is:

```python
def your_algorithm(d_single, p_bs):
    ...
```

`main()` dynamically reads the number of users from the input data:

```python
num_user = d_hat.shape[1]
```

The returned result has shape:

```python
p_hat.shape == (2, num_user)
```

The first row is the x-coordinate and the second row is the y-coordinate.

## Data Variables

`DH_FR1.mat` is expected to contain:

| Variable | Shape | Meaning |
|---|---:|---|
| `p` | `(2, N)` | ground-truth user positions |
| `d_hat` | `(18, N)` | RTT-based distance measurements |
| `BS_positions` | `(2, 18)` | base station coordinates |

`main.py` loads `p` only because the input file contains it, but the prediction logic does not use it.  
Inference uses only `d_hat`, `BS_positions`, and `model.pkl`.

## Notes

- Keep `main.py`, `model.pkl`, and `DH_FR1.mat` in the same folder.
- Do not hard-code the number of users.
- Do not include experiment scripts, diagnostic Excel files, split files, or temporary logs in the final submission.
