from typing import cast
import pandas as pd
import matplotlib.pyplot as plt
import pyterrier as pt
from ir_measures import AP, P, nDCG, IPrec
import argparse
import os
import re

if not pt.java.started():
    pt.java.init()


PIPELINE = "Stopwords,PorterStemmer"
OUTPUT_DIR = "output"

MODELS: list[tuple[str, float | None]] = [
    ("TF_IDF",       None),
    ("BM25",         None),
    ("Hiemstra_LM",  0.2),
    ("Hiemstra_LM",  0.5),
    ("Hiemstra_LM",  0.9),
    ("DirichletLM",  50.0),
    ("DirichletLM",  500.0),
    ("DirichletLM",  2500.0),
]

def read_corpus(filepath: str):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    for doc in re.findall(r"<DOC>(.*?)</DOC>", content, re.DOTALL):
        docno_match = re.search(r"<DOCNO>\s*(.*?)\s*</DOCNO>", doc)

        if not docno_match:
            continue

        text = re.sub(r"<DOCNO>.*?</DOCNO>", "", doc, flags=re.DOTALL)
        
        yield {
            "docno": docno_match.group(1).strip(),
            "text": text.strip(),
        }


def read_queries(filepath: str, max_queries: int | None = None) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    rows = []
    for top in re.findall(r"<top>(.*?)</top>", content, re.DOTALL):
        num_match   = re.search(r"<num>\s*(\d+)\s*</num>", top)
        title_match = re.search(r"<title>\s*(.*?)\s*</title>", top, re.DOTALL)
        if not num_match or not title_match:
            continue
        rows.append({"qid": num_match.group(1).strip(),
                     "query": title_match.group(1).strip()})
        if max_queries and len(rows) >= max_queries:
            break
    print(f"  {len(rows)} queries cargadas")
    return pd.DataFrame(rows)


def read_qrels_df(filepath: str, valid_qids: set[str]) -> pd.DataFrame:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docno, relevance = parts[0], parts[1], parts[2], parts[3]
            if qid not in valid_qids:
                continue
            rows.append({"qid": qid, "docno": docno, "label": int(relevance)})
    df = pd.DataFrame(rows)
    print(f"  {len(df['qid'].unique())} queries con qrels cargadas")
    return df


def indexer(filepath: str, force_reindex: bool = False):
    index_path = os.path.join(os.getcwd(), OUTPUT_DIR, "index")
    data_properties = os.path.join(index_path, "data.properties")
    assert pt.IndexFactory is not None
    if os.path.exists(data_properties) and not force_reindex:
        return pt.IndexFactory.of(index_path)
    os.makedirs(index_path, exist_ok=True)
    idx = pt.IterDictIndexer(
        index_path,
        meta={"docno": 100},
        verbose=True,
        overwrite=True,
        properties={"termpipelines": PIPELINE},
    )
    assert idx.index is not None
    return pt.IndexFactory.of(idx.index(read_corpus(filepath)))


def retriever(topics: pd.DataFrame, qrels: pd.DataFrame, model: str, smooth: float | None = None):
    # nombre único para el output
    if smooth is not None:
        run_name = f"{model}_s{smooth}"
    else:
        run_name = model

    out_dir = os.path.join(os.getcwd(), OUTPUT_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    index_path    = os.path.join(os.getcwd(), OUTPUT_DIR, "index")
    terrier_index = pt.terrier.TerrierIndex(index_path)

    if model == "TF_IDF":
        br = terrier_index.tf_idf(num_results=1000)
    elif model == "BM25":
        br = terrier_index.bm25(num_results=1000)
    elif model == "Hiemstra_LM":
        assert smooth is not None, "Hiemstra_LM requiere un valor de suavizado (Lambda)"
        br = terrier_index.hiemstra_lm(Lambda=smooth, num_results=1000)
    elif model == "DirichletLM":
        assert smooth is not None, "DirichletLM requiere un valor de suavizado (mu)"
        br = terrier_index.dirichlet_lm(mu=smooth, num_results=1000)
    else:
        raise ValueError(f"Modelo desconocido: {model}")

    recall_levels = [round(i / 10, 1) for i in range(11)]
    eval_metrics  = [AP, P @ 10, nDCG @ 10, *[IPrec @ r for r in recall_levels]]

    aggregate_df = cast(pd.DataFrame, pt.Experiment(
        [br], topics, qrels,
        eval_metrics=eval_metrics,
        names=[run_name],
    ))
    perquery_df = cast(pd.DataFrame, pt.Experiment(
        [br], topics, qrels,
        eval_metrics=eval_metrics,
        names=[run_name],
        perquery=True,
    ))

    _save_global_metrics(aggregate_df, run_name, out_dir)
    _save_rp_curve(perquery_df, run_name, recall_levels, out_dir)
    _save_per_query_distribution(perquery_df, run_name, out_dir)


def _save_global_metrics(aggregate_df: pd.DataFrame, model: str, out_dir: str):
    row = aggregate_df[aggregate_df["name"] == model].iloc[0]

    metrics = {str(AP): "MAP", str(P @ 10): "P@10", str(nDCG @ 10): "nDCG@10"}
    lines = ["Métricas globales\n", "=" * 40 + "\n"]
    for col, label in metrics.items():
        if col in row.index:
            lines.append(f"  {label:<12} {row[col]:.4f}\n")

    path = os.path.join(out_dir, "metrics_global.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("".join(lines))


def _save_rp_curve(
    perquery_df: pd.DataFrame,
    model: str,
    recall_levels: list[float],
    out_dir: str,
):
    pq = perquery_df[perquery_df["name"] == model]

    rp_points: list[tuple[float, float]] = []
    for r in recall_levels:
        col = str(IPrec @ r)
        vals = pq[pq["measure"] == col]["value"]
        rp_points.append((r, vals.mean() if not vals.empty else 0.0))

    path_txt = os.path.join(out_dir, "rp_curve.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("Curva R-P interpolada (11 puntos, promedio)\n")
        f.write("=" * 40 + "\n")
        f.write(f"  {'Recall':<10} Precision\n")
        for r, p in rp_points:
            f.write(f"  {r:<10.1f} {p:.4f}\n")

    recalls, precisions = zip(*rp_points)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recalls, precisions, marker="o", linewidth=1.8, color="#1d9e75")
    ax.fill_between(recalls, precisions, alpha=0.12, color="#1d9e75")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Curva R-P — {model}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rp_curve.png"), dpi=150)
    plt.close(fig)


def _save_per_query_distribution(
    perquery_df: pd.DataFrame, model: str, out_dir: str
):
    pq = perquery_df[perquery_df["name"] == model]

    target_measures = {
        str(AP):       "AP",
        str(P @ 10):   "P@10",
        str(nDCG @ 10): "nDCG@10",
    }

    path = os.path.join(out_dir, "metrics_per_query.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Distribución de métricas por consulta\n")
        f.write("=" * 60 + "\n\n")

        for measure_key, label in target_measures.items():
            subset = pq[pq["measure"] == measure_key].sort_values("qid", key=lambda s: s.astype(int))

            if subset.empty:
                print(f"  ⚠ no se encontró measure '{measure_key}' para {label}")
                continue

            f.write(f"{'─' * 60}\n")
            f.write(f"Métrica: {label}\n{'─' * 60}\n")
            f.write(f"  {'qid':<10} valor\n")
            for _, row in subset.iterrows():
                f.write(f"  {row['qid']:<10} {row['value']:.4f}\n")

            desc = subset["value"].describe()
            f.write(f"\n  Distribución:\n")
            for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
                f.write(f"    {stat:<8} {desc[stat]:.4f}\n")
            f.write("\n")

    pq.to_csv(os.path.join(out_dir, "metrics_per_query.csv"), index=False)
    print(f"Distribución guardada en {out_dir}/metrics_per_query.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("queries")
    parser.add_argument("qrels")
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    index = indexer(args.corpus, args.reindex)
    if index is None:
        print("Error: no se pudo generar el índice.")
        return

    topics = read_queries(args.queries)
    qrels  = read_qrels_df(args.qrels, set(topics["qid"]))
    for model, smooth in MODELS:
        label = f"{model}_s{smooth}" if smooth is not None else model
        print(f"Procesando: {label}")
        retriever(topics, qrels, model=model, smooth=smooth)

if __name__ == "__main__":
    main()