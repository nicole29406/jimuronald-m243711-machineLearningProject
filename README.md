# Zimbabwe Student Dropout — Train & Predict

End-to-end pipeline that trains three classifiers on the Zimbabwe student
dropout dataset and serves predictions through a Streamlit web app.

- **Models:** Logistic Regression, K-Nearest Neighbours, Gaussian Naive Bayes
- **Notebook:** `train_models.ipynb` (run from VS Code)
- **App:** `app.py` (Streamlit, runs on `http://localhost:8502`)
- **Data:** `C:\Users\nicky\Documents\zimbabwe_student_dropout (1).csv` (523,574 rows)

## Folder layout

```
model/
├── train_models.ipynb     ← train the 3 models (run first)
├── app.py                 ← Streamlit prediction UI
├── run_app.bat            ← double-click launcher (port 8502)
├── requirements.txt
├── README.md
└── artifacts/             ← created by the notebook
    ├── logreg.joblib
    ├── knn.joblib
    ├── nb.joblib
    ├── feature_columns.json
    └── metrics.json
```

## How to run (VS Code)

1. Open this folder in VS Code (`File ▶ Open Folder…`).
2. (Optional) install the recommended extensions: **Python**, **Jupyter**.
3. (Optional) install dependencies if any are missing:
   ```
   pip install -r requirements.txt
   ```
4. Open `train_models.ipynb`, pick the Python 3.12 kernel (top right),
   and click **Run All**. Training takes a few minutes — KNN is the slowest
   step. When done, the `artifacts/` folder appears.
5. Launch the Streamlit app from a VS Code terminal:
   ```
   streamlit run app.py --server.port 8502
   ```
   or just double-click **`run_app.bat`**.
6. The browser opens at **http://localhost:8502** — fill in the form and
   click **Predict dropout risk**.

## Using the app

- Pick a single model in the sidebar, or choose **Compare all 3** to see
  every model's prediction and probability side by side.
- Hit **Autofill from a real CSV row** to populate the form with real data
  (faster than typing 14 fields).
- Test-set accuracy / precision / recall / F1 for each model is shown in
  the sidebar.

## Notes

- The Streamlit port is **8502**, not the default 8501, so it doesn't clash
  with other Streamlit apps you might already be running.
- KNN is evaluated in the notebook on a 20,000-row sample of the test set
  because predicting on the full 105k test set with 419k stored neighbours
  is slow. Single-row predictions in the app are still fast.
- Missing values in `family_income`, `study_hours`, `attendance_rate` are
  median-imputed by the pipeline; missing `fees_paid` is treated as `"None"`.
