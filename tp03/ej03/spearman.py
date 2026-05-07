from scipy import stats
import pandas as pd
import os

def spearman_entre_modelos(run_a: str, run_b: str) -> None:
    def load_ap(run: str) -> pd.Series:
        path = os.path.join("output", run, "metrics_per_query.csv")
        df = pd.read_csv(path)
        return df[df["measure"] == "AP"].set_index("qid")["value"]

    ap_a = load_ap(run_a)
    ap_b = load_ap(run_b)

    result = stats.spearmanr(ap_a, ap_b)
    rho = float(result.correlation)  # type: ignore[union-attr]
    p   = float(result.pvalue)       # type: ignore[union-attr]
    print(f"Spearman ρ ({run_a} vs {run_b}): {rho:.4f}  (p={p:.4f})")

if __name__ == "__main__":
    spearman_entre_modelos("TF_IDF", "Hiemstra_LM_s0.9")
    spearman_entre_modelos("TF_IDF", "DirichletLM_s50.0")
    spearman_entre_modelos("Hiemstra_LM_s0.9", "DirichletLM_s50.0")