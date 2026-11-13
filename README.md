# Review sentiment analysis experiments

Two notebook-exported Python experiments explore sentiment in product reviews using NLTK VADER and a pretrained RoBERTa model.

## Files

- `SN-ANALYSIS.py` explores review scores and VADER sentiment.
- `SN_ANALYSIS_PRE_TRAINED (1).py` loads a pretrained sentiment model, attempts a VADER/RoBERTa comparison on a sample, and plots results.

## Data and dependencies

Both scripts expect a local `Reviews.csv`, which is not included in this repository. The scripts use columns including `Text`, `Id`, and `Score`; the pretrained-model experiment samples 500 rows. Dependencies include pandas, NumPy, Matplotlib, seaborn, NLTK, and notebook-oriented tqdm; the second experiment also uses Transformers, PyTorch, and SciPy. NLTK data and the pretrained model may need to be obtained separately.

## Current limitations

These files retain notebook-exported code and have not been verified as standalone programs. The pretrained-model script refers to `sia` without initializing it, so its combined analysis cannot run correctly as written. Obtain the data and review the scripts and dependencies before use.
