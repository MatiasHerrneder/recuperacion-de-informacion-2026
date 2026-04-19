from typing import cast
import pandas as pd
import pyterrier as pt
import argparse
import os
from scipy.stats import spearmanr


if not pt.java.started():
    pt.java.init()


PIPELINE = "Stopwords,PorterStemmer"
OUTPUT_DIR = "output"
MODELS = ["TF_IDF", "BM25"]


def read_files(dir: str):
    """Genera dicts con docno y text por cada archivo"""
    for filepath in pt.io.find_files(dir):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        yield {
            "docno": filepath,
            "text": text
        }

def indexer(dir: str, force_reindex: bool = False):
    try:
        index_path = os.path.join(os.getcwd(), OUTPUT_DIR, "index")
        data_properties = os.path.join(index_path, "data.properties")

        if os.path.exists(data_properties) and not force_reindex:
            assert pt.IndexFactory is not None
            return pt.IndexFactory.of(index_path)
        
        os.makedirs(index_path, exist_ok=True)

        indexer = pt.IterDictIndexer(
            index_path,
            meta={'docno': 100},
            verbose=True,
            overwrite=True,
            properties={"termpipelines": PIPELINE},
        )

        assert indexer.index is not None
        indexref = indexer.index(read_files(dir))

        assert pt.IndexFactory is not None
        return pt.IndexFactory.of(indexref)
    
    except Exception as e:
        print(f"Error al generar el indice: {e}")
        return None

def retriever(index, queries, model):
    try:
        out_dir = os.path.join(os.getcwd(), OUTPUT_DIR, "queries", model)

        assert pt.terrier.Retriever is not None
        br = pt.terrier.Retriever(index, 
            wmodel=model,  
            metadata=['docno','filename'],
            properties={"termpipelines": PIPELINE},
        ) 

        os.makedirs(out_dir, exist_ok=True)
        results = {}
        for i, query in enumerate(queries):
            query_res = cast(pd.DataFrame, br.search(query))
            results[query] = []
            with open(os.path.join(out_dir, f"Q{i}.txt"), "w", encoding="utf-8") as f:
                for _, row in query_res.iterrows():
                    f.write(f"{row['docno']}\t{row['filename']}\n")
                    results[query].append(row['docid'])

        return results

    except Exception as e:
        print(f"Error al realizar la consulta: {e}")

def model_correlation(results):
    try:
        top_k = [10, 25, 50]

        with open(os.path.join(OUTPUT_DIR, "correlation.txt"), "w", encoding="utf-8") as f:
                model1, model2 = results.keys()
                for query in results[model1].keys():
                    f.write(f"QUERY: {query}\n")
                    for k in top_k:
                        correlation, _ = spearmanr(results[model1][query][:k], results[model2][query][:k])
                        f.write(f"Coeff. de correlación [{model1}, {model2}], top-{k}: {correlation:.4f}\n")
                        # correlation, p_value = spearmanr(results[model1][query][:k], results[model2][query][:k])
                        # f.write(f"Coeff. de correlación [{model1}, {model2}], top-{k}: {correlation:.4f} (p-value: {p_value:.4f})\n")

    except Exception as e:
        print(f"Error al calcular la correlación: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with the files to index (scans recursively)")
    parser.add_argument("queries", help="Queries to search (separated by commas)")
    parser.add_argument("--reindex", action="store_true", help="Forces reindexing if it exists")
    args = parser.parse_args()

    index = indexer(args.dir, args.reindex)

    if index is not None:
        results = {}
        for model in MODELS:
            results[model] = retriever(index, queries=args.queries.split(","), model=model)

        model_correlation(results)

if __name__ == "__main__":
    main()