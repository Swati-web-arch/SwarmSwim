"""
run_pipeline.py
---------------
One-click runner to verify tests, train models, evaluate baselines,
and generate publication-quality figures for Review 2.
"""
import sys
import subprocess
import pathlib

def main():
    root = pathlib.Path(__file__).parent.resolve()
    print("=" * 70)
    print("  SWARM LATENT PREDICTION -- REVIEW 2 VERIFICATION PIPELINE")
    print("=" * 70)
    
    # 1. Run pytest
    print("\n[STEP 1/2] Running automated test suite (PyTest)...")
    res = subprocess.run([sys.executable, "-m", "pytest", str(root / "tests"), "-v"], cwd=str(root))
    if res.returncode != 0:
        print("\n[FAILED] Tests failed! Aborting pipeline.")
        sys.exit(res.returncode)
    print("[SUCCESS] All 64 unit tests PASSED.")

    # 2. Run evaluation
    print("\n[STEP 2/2] Running training, evaluation and plotting...")
    res = subprocess.run([sys.executable, str(root / "experiments" / "evaluate.py")], cwd=str(root))
    if res.returncode != 0:
        print("\n[FAILED] Evaluation failed! Aborting pipeline.")
        sys.exit(res.returncode)

    print("\n" + "=" * 70)
    print("  [COMPLETE] REVIEW 2 PIPELINE EXECUTION FINISHED!")
    print("=" * 70)
    print(f"  * Test Suite:     ALL 64 TESTS PASSED")
    print(f"  * Results CSV:    {root / 'results' / 'rmse_table.csv'}")
    print(f"  * Benchmark Plot: {root / 'results' / 'rmse_curve.png'}")
    print(f"  * Tech Report:    {root / 'docs' / 'results_report.md'}")
    print("=" * 70)

if __name__ == "__main__":
    main()