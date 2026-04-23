from typing import cast
import pandas as pd
import pyterrier as pt
import argparse
import os
import re
import shutil
import math


if not pt.java.started():
    pt.java.init()


PIPELINE = "Stopwords,PorterStemmer"
OUTPUT_DIR = "output"
MODEL = "TF_IDF"
MAX_QUERIES = 12
RESULTS_PER_QUERY = 10


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

def read_queries(filepath: str, max_queries: int | None = None) -> list[dict]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    queries = []
    for top in re.findall(r"<top>(.*?)</top>", content, re.DOTALL):
        num_match = re.search(r"<num>\s*(\d+)\s*</num>", top)
        title_match = re.search(r"<title>\s*(.*?)\s*</title>", top, re.DOTALL)

        if not num_match or not title_match:
            continue

        queries.append({
            "qid": num_match.group(1).strip(),
            "query": title_match.group(1).strip(),
        })

        if max_queries and len(queries) >= max_queries:
            break

    print(f"  {len(queries)} queries cargadas (máximo: {max_queries})")
    return queries

def read_qrels(filepath: str, valid_qids: set[str]) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            qid, _, docno, relevance = parts[0], parts[1], parts[2], parts[3]

            if qid not in valid_qids:
                continue

            if int(relevance) > 0:
                qrels.setdefault(qid, set()).add(docno)

    print(f"  {len(qrels)} queries con qrels cargadas")
    return qrels

def indexer(filepath: str, force_reindex: bool = False):
    try:
        index_path = os.path.join(os.getcwd(), OUTPUT_DIR, "index")
        data_properties = os.path.join(index_path, "data.properties")

        if os.path.exists(data_properties) and not force_reindex:
            assert pt.IndexFactory is not None
            return pt.IndexFactory.of(index_path)

        os.makedirs(index_path, exist_ok=True)

        indexer = pt.IterDictIndexer(
            index_path,
            meta={"docno": 100},
            verbose=True,
            overwrite=True,
            properties={"termpipelines": PIPELINE},
        )

        assert indexer.index is not None
        indexref = indexer.index(read_corpus(filepath))

        assert pt.IndexFactory is not None
        return pt.IndexFactory.of(indexref)

    except Exception as e:
        print(f"Error al generar el indice: {e}")
        return None


def retriever(index, queries: list[dict], qrels: dict[str, set[str]], results_per_query: int | None = None, model: str = MODEL):
    try:
        out_dir = os.path.join(os.getcwd(), OUTPUT_DIR, "queries", model)

        # limpiar carpeta
        if os.path.exists(out_dir):
            for item in os.listdir(out_dir):
                ruta = os.path.join(out_dir, item)
                if os.path.isfile(ruta) or os.path.islink(ruta):
                    os.remove(ruta)
                elif os.path.isdir(ruta):
                    shutil.rmtree(ruta)
        else:
            os.makedirs(out_dir, exist_ok=True)

        assert pt.terrier.Retriever is not None
        br = pt.terrier.Retriever(
            index,
            wmodel=model,
            metadata=["docno"],
            properties={"termpipelines": PIPELINE},
        )

        metrics = {
            "MAP": 0.0,
            "mean_P@10": 0.0,
            "mean_nDCG@10": 0.0,
            "avg_precision": [],
            "avg_recall": [],
            "avg_rp_11": [0.0] * 11
        }

        all_precision = {}
        all_recall = {}

        recall_levels = [i / 10 for i in range(11)]

        for q in queries:
            qid = q["qid"]
            query_text = q["query"]
            relevant_docs = qrels.get(qid, set())

            query_res = cast(pd.DataFrame, br.search(query_text))

            retrieved_relevant = 0
            avg_precision = 0
            dcg_at_10 = 0

            precision_q = []
            recall_q = []

            out_file = os.path.join(out_dir, f"Q{qid}.txt")

            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"Query {qid}: {query_text}\n")
                f.write(f"Total de relevantes: {len(relevant_docs)}\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Rank':<6} {'Docno':<12} {'Score':<12} {'Relevante'}\n")
                f.write("-" * 60 + "\n")

                for rank, (_, row) in enumerate(query_res.iterrows(), start=1):
                    if results_per_query and rank > results_per_query:
                        break

                    docno = str(row["docno"])
                    score = row.get("score", 0.0)
                    is_relevant = docno in relevant_docs

                    if is_relevant:
                        retrieved_relevant += 1
                        avg_precision += retrieved_relevant / rank

                    rel = 1 if is_relevant else 0

                    # DCG@10
                    if rank <= 10:
                        if rank == 1:
                            dcg_at_10 += rel
                        else:
                            dcg_at_10 += rel / math.log2(rank)

                    precision = retrieved_relevant / rank
                    recall = retrieved_relevant / len(relevant_docs) if len(relevant_docs) > 0 else 0

                    precision_q.append(precision)
                    recall_q.append(recall)

                    f.write(f"{rank:<6} {docno:<12} {score:<12.4f} {'1' if is_relevant else '0'}\n")

                # ===== AP =====
                AP = avg_precision / len(relevant_docs) if len(relevant_docs) > 0 else 0

                # ===== P@10 =====
                k = min(10, len(precision_q))
                P_at_10 = precision_q[k - 1] if k > 0 else 0

                # ===== nDCG@10 =====
                ideal_hits = min(len(relevant_docs), 10)
                idcg = 0
                for i in range(1, ideal_hits + 1):
                    if i == 1:
                        idcg += 1
                    else:
                        idcg += 1 / math.log2(i)

                nDCG_10 = dcg_at_10 / idcg if idcg > 0 else 0

                # ===== curva R-P (11 puntos) =====
                rp_11 = []
                for r_level in recall_levels:
                    precisions = [
                        p for p, r in zip(precision_q, recall_q) if r >= r_level
                    ]
                    rp_11.append(max(precisions) if precisions else 0)

                for i in range(11):
                    metrics["avg_rp_11"][i] += rp_11[i]

                # ===== escribir métricas =====
                f.write("\n" + "=" * 60 + "\n")
                f.write("Métricas:\n")
                f.write(f"P@10: {P_at_10:.4f}\n")
                f.write(f"AP: {AP:.4f}\n")
                f.write(f"nDCG@10: {nDCG_10:.4f}\n")

                f.write("\nCurva R-P (11 puntos):\n")
                for i, val in enumerate(rp_11):
                    f.write(f"Recall {i/10:.1f}: {val:.4f}\n")

                # formato lista (excel-friendly)
                f.write("\nRP_11 (lista):\n")
                f.write(", ".join(str(x) for x in rp_11) + "\n")

            # ===== acumular globales =====
            metrics["MAP"] += AP
            metrics["mean_P@10"] += P_at_10
            metrics["mean_nDCG@10"] += nDCG_10

            all_precision[qid] = precision_q
            all_recall[qid] = recall_q

        # ===== promedios =====
        n = len(queries)
        metrics["MAP"] /= n
        metrics["mean_P@10"] /= n
        metrics["mean_nDCG@10"] /= n
        for i in range(11):
            metrics["avg_rp_11"][i] /= n

        max_rank = max(len(v) for v in all_precision.values())

        for i in range(max_rank):
            p_values = [v[i] for v in all_precision.values() if i < len(v)]
            r_values = [v[i] for v in all_recall.values() if i < len(v)]

            metrics["avg_precision"].append(sum(p_values) / len(p_values))
            metrics["avg_recall"].append(sum(r_values) / len(r_values))

        # ===== guardar métricas globales =====
        with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write("Metricas globales:\n")
            f.write(f"MAP: {metrics['MAP']:.4f}\n")
            f.write(f"Mean P@10: {metrics['mean_P@10']:.4f}\n")
            f.write(f"Mean nDCG@10: {metrics['mean_nDCG@10']:.4f}\n")
            f.write("\nPrecision promedio por rank:\n")

            f.write(", ".join(str(x) for x in metrics["avg_precision"]) + "\n")

            f.write("\nRecall promedio por rank:\n")
            f.write(", ".join(str(x) for x in metrics["avg_recall"]) + "\n")

            f.write("\nCurva R-P interpolada promedio (11 puntos):\n")
            for i, val in enumerate(metrics["avg_rp_11"]):
                f.write(f"Recall {i/10:.1f}: {val:.4f}\n")

            f.write("\nRP_11 promedio (lista):\n")
            f.write(", ".join(str(x) for x in metrics["avg_rp_11"]) + "\n")

        print(metrics)

    except Exception as e:
        print(f"Error al realizar la consulta: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Archivo .trec con el corpus")
    parser.add_argument("queries", help="Archivo con queries en formato TREC")
    parser.add_argument("qrels", help="Archivo con qrels (qid 0 docno relevance)")
    # parser.add_argument("--max-queries", type=int, default=MAX_QUERIES, help=f"Máximo de queries a procesar (default: {MAX_QUERIES})")
    parser.add_argument("--reindex", action="store_true", help="Fuerza re-indexado")
    args = parser.parse_args()
    
    index = indexer(args.corpus, args.reindex)

    if index is None:
        print("Error: no se pudo generar el índice.")
        return

    queries = read_queries(args.queries)
    # queries = read_queries(args.queries, max_queries=MAX_QUERIES)
    valid_qids = {q["qid"] for q in queries} 
    qrels = read_qrels(args.qrels, valid_qids)

    retriever(index, queries, qrels)

if __name__ == "__main__":
    main()