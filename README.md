# Machine Learning Assignment 2

## a. Problem statement

An institute wants to know, from information available during the first year, whether a student will drop out, stay enrolled, or graduate. This is a three-class classification problem. Missing a student who is about to leave is costly, and so is treating every student as a future graduate. The work trains five classifiers on one public dataset, scores them on a held-out test set, and serves those saved models in a Streamlit app. The app does not refit a model when a file is uploaded.

## b. Dataset description

Source: Realinho, V., Vieira Martins, M., Machado, J., and Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

The file used here has **4,424 students and 36 input features**, which meets the assignment minimum of 500 rows and 12 features. There are no missing values. The target column is `Target`, with three labels:

| Class | Students | Share |
| --- | ---: | ---: |
| Graduate | 2,209 | 49.9% |
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |

The features cover demographics and application data (marital status, application mode and order, course, attendance, previous qualification, nationality, parents' qualification and occupation, admission grade, age), enrolment flags (displaced, special educational needs, debtor, tuition up to date, gender, scholarship, international student), first- and second-semester curricular results (credited, enrolled, evaluated, approved, grade, and units without evaluation), and three macroeconomic indicators (unemployment rate, inflation rate, GDP). Course, nationality, qualification, and occupation are stored as integer codes. Those codes are category labels, not magnitudes.

Preprocessing, fit on the training split only and saved inside each pipeline:

- Logistic regression, decision tree, k-nearest neighbours, and the support vector machine: one-hot encode the nine nominal code columns (`handle_unknown="ignore"`) and standardise the numeric and binary columns.
- Gaussian naive Bayes: standardise the original 36 columns and do not one-hot encode them. One-hot indicators are not Gaussian, and a Gaussian naive Bayes model fit on them is not a meaningful use of that classifier.

Split: stratified 80/20, `random_state=42`. Training rows: 3,539. Test rows: 885, saved as `test_data.csv` with the 36 features and `Target`. Reported metrics are computed on this test set only.

## c. Github Repository Link

https://github.com/YOUR_USERNAME/academic-outcome-classifier

Replace `YOUR_USERNAME` with the public repository URL before this README is copied into the submission PDF.

The repository contains:

- `app.py` — Streamlit app. Loads the saved pipelines and scores an uploaded CSV. It does not call `fit`.
- `train.py` — trains all five models and writes `test_data.csv` and `model/`.
- `evaluation.py` — the metric definitions shared by training and the app.
- `schema.py` — feature roles and model filenames.
- `requirements.txt`
- `README.md`
- `test_data.csv` — the held-out test set used for the table below.
- `model/` — `logistic_regression.joblib`, `decision_tree.joblib`, `knn.joblib`, `naive_bayes.joblib`, `svm.joblib`, and `training_record.joblib`.
- `data/dataset.csv` — the full UCI file, so `train.py` can be rerun.

## d. Models used

All five models are trained on the same students and scored on the same 885 test rows.

Settings:

- Logistic regression: L2, `C=1`, `lbfgs`, `class_weight="balanced"`, `max_iter=2000`.
- Decision tree: Gini, `max_depth=8`, `min_samples_leaf=15`, `class_weight="balanced"`.
- K-nearest neighbours: `k=15`, distance weights, Euclidean distance, after scaling.
- Naive Bayes: `GaussianNB` (`var_smoothing=1e-9`) on the standardised original columns.
- Support vector machine: RBF kernel, `C=2`, `gamma="scale"`, `class_weight="balanced"`, with probability estimates so ROC AUC can be computed.

Precision, recall, and F1 are **macro** averages over Dropout, Enrolled, and Graduate, so each class counts equally. AUC is the **weighted one-vs-rest ROC AUC** from `predict_proba`. MCC is the multiclass Matthews correlation. A macro average is used because weighted recall on a single-label problem is identical to accuracy, and the two columns would then not be separate measurements.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7243 | 0.8965 | 0.6956 | 0.7000 | 0.6876 | 0.5729 |
| Decision Tree | 0.6633 | 0.8585 | 0.6668 | 0.6633 | 0.6401 | 0.5044 |
| KNN | 0.7232 | 0.8548 | 0.6756 | 0.6124 | 0.6235 | 0.5403 |
| Naive Bayes | 0.6588 | 0.8118 | 0.5675 | 0.5579 | 0.5576 | 0.4279 |
| Support Vector Machine | 0.7367 | 0.8971 | 0.6991 | 0.7023 | 0.6952 | 0.5833 |

The test set has 284 Dropout, 159 Enrolled, and 442 Graduate rows. A classifier that always predicted Graduate would reach accuracy 442/885 = 0.4994 and would score poorly on macro F1. Every model above beats that accuracy.

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 0.7243 and AUC 0.8965. Balanced class weights keep the small Enrolled class visible: Enrolled recall is 0.6352 (101 of 159) and Graduate recall is 0.7851. Enrolled precision is only 0.4156, so many of the students labelled Enrolled are actually in another class. This is the strongest linear model and its AUC is 0.0006 below the SVM. |
| Decision Tree | Accuracy 0.6633, macro F1 0.6401, MCC 0.5044. The depth cap of 8 and the leaf size of 15 stop the tree from memorising course codes. It has the highest Enrolled recall, 0.6918, because balanced class weights push splits toward that class. The cost is Graduate recall of 0.6855, against 0.8122 for the SVM. AUC 0.8585 shows the leaf probabilities still rank students usefully. |
| KNN | Accuracy 0.7232 is close to logistic regression, but macro recall is 0.6124 and macro F1 is 0.6235. Enrolled recall is 0.2579. With k=15 in a one-hot space, most neighbours are Graduates, so the vote drifts to the majority: Graduate recall is 0.9525, the highest of the five models. Accuracy stays high for that reason, while MCC (0.5403) shows a weaker association than logistic regression or the SVM. |
| Naive Bayes | Accuracy 0.6588, AUC 0.8118, macro F1 0.5576, MCC 0.4279, the lowest of the five. Course, qualification, and occupation codes are not Gaussian, and first-semester results move with second-semester results, so the independence assumption does not hold. Enrolled recall is 0.1824 and Enrolled precision is 0.2736. AUC still shows ranking signal above chance. The model is the right Gaussian naive Bayes for these columns; the score is the cost of that assumption on this table. |
| Support Vector Machine | Best value in every column: accuracy 0.7367, AUC 0.8971, macro precision 0.6991, macro recall 0.7023, macro F1 0.6952, MCC 0.5833. The RBF kernel picks up interactions between academic load and grades that a linear boundary misses. Dropout recall is 0.6972 and Graduate recall is 0.8122. Enrolled is still the hard class (recall 0.5975, precision 0.4419), and its F1 of 0.5080 is the best Enrolled F1 in the comparison. |
| Overall, Winner for your dataset? | Support vector machine. It leads accuracy, AUC, macro precision, macro recall, macro F1, and MCC on the same 885 test students. Logistic regression is a close second on AUC. K-nearest neighbours matches logistic regression on accuracy only by over-predicting Graduate. The decision tree finds more Enrolled students and loses more Graduates. Gaussian naive Bayes is last because the coded categorical fields and the paired semester grades violate its assumptions. |

## Streamlit app

The deployed app should open on the bundled test set and show the table above without an upload. An upload replaces that file. The CSV must contain the 36 feature columns and `Target` (`Dropout`, `Enrolled`, or `Graduate`). A comma-separated or semicolon-separated file is accepted. The selected model, the six metrics, a confusion matrix, and a classification report are shown for the uploaded rows. A second table scores all five saved models on that same file. No model is trained in the app.

Deploy on Streamlit Community Cloud from this repository, branch `main`, main file `app.py`. In Advanced settings, set the Python version to **3.13**. The free-tier file to upload, if you want to exercise the control, is `test_data.csv`.

## Reproduce

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

`python train.py` prints the comparison table. That command is the run to capture on BITS Virtual Lab.
