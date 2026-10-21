# ML Lab Explorer

Interactive Node.js website for the supplied notebook datasets:
- **Clustering:** airline `train.csv` and `test.csv` (K-Means, cluster sizes, inertia, 2-feature scatter plot).
- **Regression:** Solar Plant 1 and Plant 2 generation/weather CSVs (multiple linear regression, R², RMSE, MAE, actual-vs-predicted chart).
- **Classification:** machine failure dataset (binary logistic regression, accuracy/precision/recall/F1, confusion counts).

## Run in VS Code (Windows PowerShell)

1. Copy this `ml_explorer_app` folder into your existing `ml_caseStudy` repository folder (next to `data` and `notebooks`), or copy its contents into the repository.
2. Open a terminal in the `ml_explorer_app` folder.
3. Run:
   ```powershell
   npm install
   npm start
   ```
4. Open http://localhost:3000 in your browser.

Requires Node.js 18+.

## Git: commit the new app

Open a terminal in the **repository root** (`ml_caseStudy`, not inside a nested Git repository) and run:
```powershell
git status
git add ml_explorer_app
git commit -m "Add interactive ML Lab Explorer"
git push origin main
```
If you copied the app's files directly into the repository root instead of a subfolder, use `git add .` after checking `git status`.

## Notes
- The app uses the actual CSV files copied into `data/`; it does not change your notebooks.
- It implements transparent baseline algorithms in JavaScript, not all of the scikit-learn models in the notebooks. Use your notebooks for full model comparisons/tuning and research-grade evaluation.
- For responsiveness, regression uses up to 8,000 complete rows, classification up to 10,000, and clustering up to 15,000.
- The app does not automatically commit or push; review changes and run the Git commands yourself.
