# Product Recommendation Pipeline

This project predicts whether a customer recommends a women's clothing product. The complete scikit-learn pipeline handles numeric, categorical, and review text/title features, engineers review length, and tunes logistic-regression regularization with cross-validation.

## Run

```powershell
python -m pip install -r requirements.txt
python solution.py
```

`Data_Science_Pipeline.ipynb` is the executed notebook deliverable. `solution.py` contains the reusable pipeline and training function. The included `data/reviews.csv` has 18,442 anonymized reviews.
