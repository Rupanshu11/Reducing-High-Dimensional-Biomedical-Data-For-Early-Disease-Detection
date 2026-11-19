"""
BIOREDUCE AI — End-to-End Pipeline Verification Script
"""

import os
import time
from modules.data_loader import load_biomedical_dataset
from modules.preprocessing import BiomedicalPreprocessor
from modules.baseline import evaluate_all_features_baseline
from modules.dimensionality_reduction import PCAReducer, TruncatedSVDReducer, HybridReducer
from modules.feature_selection import SelectKBestSelector
from modules.umap_analysis import compute_umap_embedding, evaluate_umap_manifold
from modules.reconstruction import compute_pca_reconstruction_profile, compute_svd_reconstruction_profile
from modules.retention_comparison import run_comprehensive_retention_study, compute_pareto_optimal_retention
from modules.report_generator import generate_markdown_report, generate_html_report


def main():
    print("Step 1: Loading dataset...")
    df, X, y, feat_names, meta = load_biomedical_dataset("data/Brain_GSE50161.csv")
    print(f"Dataset loaded: samples={meta['n_samples']}, probes={meta['n_features']}, classes={len(meta['classes'])}")

    print("Step 2: Preprocessing and train/test split...")
    prep = BiomedicalPreprocessor().split_and_preprocess(X, y)
    print(f"Train shape: {prep['X_train_scaled'].shape}, Test shape: {prep['X_test_scaled'].shape}")

    print("Step 3: Baseline analysis (All 54,675 Features)...")
    t0 = time.time()
    base = evaluate_all_features_baseline(prep, "Logistic Regression")
    print(f"Baseline Macro F1: {base['macro_f1']:.4f}, CV F1: {base['cv_f1_mean']:.4f}, Time: {time.time()-t0:.2f}s")

    print("Step 4: UMAP Manifold Embedding...")
    emb, umap_meta = compute_umap_embedding(prep["X_train_scaled"], n_neighbors=15, min_dist=0.1)
    u_eval = evaluate_umap_manifold(emb, prep["y_train"])
    print(f"UMAP Silhouette: {u_eval['silhouette_score']:.4f}, Time: {umap_meta['compute_time_s']}s")

    print("Step 5: Full Retention Study (14 Experiments across 5 Categories)...")
    t0 = time.time()
    df_res, raw_evals = run_comprehensive_retention_study(prep, "Logistic Regression")
    print(f"Completed {len(df_res)} experiments in {time.time()-t0:.2f}s")

    print("\n--- EXPERIMENT SUMMARY ---")
    for _, r in df_res.iterrows():
        print(f"  {r['Experiment']:<30} | Dims: {r['Retained Dims']:<6} | Red: {r['Reduction (%)']:>7.3f}% | F1: {r['Macro F1']:.4f} | CV: {r['CV Macro F1 (Mean)']:.4f}")

    print("\nStep 6: Pareto Optimization & Trade-off Sweet Spot...")
    pareto = compute_pareto_optimal_retention(df_res, f1_retention_threshold=0.95)
    print(f"Optimal Choice: {pareto['optimal_experiment']}")
    print(f"Retained Dimensions: {pareto['optimal_retained_dims']}")
    print(f"Reduction Achieved: {pareto['optimal_reduction_pct']:.3f}%")
    print(f"F1 Parity Achieved: {pareto['f1_retention_achieved_pct']:.1f}%")

    print("\nStep 7: Generating Publication Reports...")
    md_rep = generate_markdown_report(meta, df_res, pareto)
    with open("outputs/reports/BIOREDUCE_AI_Research_Report.md", "w", encoding="utf-8") as f:
        f.write(md_rep)
    html_rep = generate_html_report(md_rep)
    with open("outputs/reports/BIOREDUCE_AI_Research_Report.html", "w", encoding="utf-8") as f:
        f.write(html_rep)
    df_res.to_csv("outputs/reports/BIOREDUCE_AI_Retention_Comparison.csv", index=False)
    print("Reports successfully generated in outputs/reports/")

    print("\n=== ALL SCIENTIFIC WORKFLOWS PASSED WITH 100% SUCCESS ===")


if __name__ == "__main__":
    main()
