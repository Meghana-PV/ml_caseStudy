ML Lab Explorer Pro
Interactive Node.js website for the provided machine-learning datasets. The UI is built with HTML/CSS/JavaScript and Chart.js. Express serves the website and API; a Python/scikit-learn engine trains and evaluates the actual notebook-selected models.

Included modules
1. Clustering
Airline train.csv and test.csv selectable separately.
K-Means and hierarchical Agglomerative Clustering (Ward linkage).
Interactive cluster count, PCA scatter plot, cluster distribution, inertia, Silhouette, Davies–Bouldin and Calinski–Harabasz metrics.
K-Means uses up to 10,000 rows; hierarchical clustering uses up to 3,000 because it is much more expensive. Dendrogram merge distances use a subset of 60 records.
2. Regression
Plant 1 or Plant 2 generation data, with timestamp-matched average weather sensor readings merged by timestamp.
Random Forest Regressor (100 trees), based on the notebook comparison where Random Forest had the lowest RMSE/MAE.
R², RMSE, MAE, actual-versus-predicted chart, feature importance, and interactive AC_POWER prediction form.
3. Classification
Machine failure dataset.
Gradient Boosting Classifier using tuned notebook parameters: n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42. It had the highest tuned weighted F1 in the notebook's final comparison table (0.988198). The notebook separately saves a tuned Random Forest model, but Gradient Boosting is the top entry in the final comparison table.
Holdout accuracy, weighted precision/recall/F1, ROC-AUC, confusion counts, feature importance, and an interactive failure prediction form.
Model is evaluated on a stratified 20% holdout and then refit on all records for predictions.
Requirements
Node.js 18+
Python 3.10+ recommended
Internet connection on first page load for Google Fonts and Chart.js CDN, or replace CDN with local copies.
Run in VS Code (Windows PowerShell)
Extract this folder and open a terminal inside ml_lab_explorer_pro.

npm install
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
npm start
Then open http://localhost:3000.

If PowerShell blocks environment activation, run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass in that terminal, then activate again. Alternatively install dependencies into your regular Python with py -m pip install -r requirements.txt.

If the python command is not found, set the environment variable PYTHON to your Python executable path before running Node, for example:

$env:PYTHON = "py"
npm start
Normally use python after activating .venv.

Files
server.js — Express server/API.
ml_engine.py — pandas/scikit-learn model engine invoked by Node.
public/ — frontend.
data/ — copies of the supplied CSV files.
requirements.txt — Python ML dependencies.
Git
From the repository root containing this folder:

git status
git add ml_lab_explorer_pro
git commit -m "Add ML Lab Explorer with clustering regression classification"
git push origin main
Notes
This is a local educational project, not a production prediction service.
The notebook regression features include DC power and yield measurements which are highly related to AC power. Very high scores should be interpreted with that relationship and possible data leakage in mind; use a time-based holdout for a more realistic future-generation forecasting evaluation.
Hierarchical clustering is computationally expensive; the UI intentionally limits its row count.
