import argparse
import os
import pickle
import statistics
import time

from tokenizer import tokenizer
from script import retrieve_TAAT, retrieve_DAAT

VOCABULARY_SUBPATH = "vocabulary.pkl"
TERM_ID_SUBPATH = "term_index.pkl"

REPETITIONS = 1


def query_length_bucket(query: str):

    terms = tokenizer(query)

    if len(terms) == 1:
        return "1 termino"

    if len(terms) <= 3:
        return "2-3 terminos"

    return "4+ terminos"


def average_posting_size(query, vocabulary, term_to_id):

    terms = tokenizer(query)

    lengths = []

    for term in terms:

        term_id = term_to_id.get(term)

        if term_id is None:
            continue

        lengths.append(
            vocabulary[term_id][1]
        )

    if not lengths:
        return 0

    return sum(lengths) / len(lengths)


def posting_bucket(avg_posting_size):

    if avg_posting_size < 5:
        return "< 5"

    if avg_posting_size < 10:
        return "5-10"

    return "10+"


def benchmark_query(index_path, query, k):

    start = time.perf_counter()

    for _ in range(REPETITIONS):
        retrieve_TAAT(
            index_path,
            query,
            k
        )

    taat_time = (
        time.perf_counter() - start
    ) / REPETITIONS

    start = time.perf_counter()

    for _ in range(REPETITIONS):
        retrieve_DAAT(
            index_path,
            query,
            k
        )

    daat_time = (
        time.perf_counter() - start
    ) / REPETITIONS

    return taat_time, daat_time


def print_group_results(title, groups):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    for bucket in sorted(groups.keys()):

        taat_times = groups[bucket]["taat"]
        daat_times = groups[bucket]["daat"]

        if not taat_times:
            continue

        taat_avg = statistics.mean(taat_times)
        daat_avg = statistics.mean(daat_times)

        print(f"\n{bucket}")
        print(f"  Queries : {len(taat_times)}")
        print(f"  TAAT    : {taat_avg:.6f} s")
        print(f"  DAAT    : {daat_avg:.6f} s")

        if taat_avg > 0:
            print(
                f"  DAAT/TAAT : {(daat_avg / taat_avg):.2f}x"
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "index_path",
        help="Directorio del indice"
    )

    parser.add_argument(
        "queries_file",
        help="Archivo con una query por linea"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10
    )

    args = parser.parse_args()

    with open(
        os.path.join(
            args.index_path,
            VOCABULARY_SUBPATH
        ),
        "rb"
    ) as f:
        vocabulary = pickle.load(f)

    with open(
        os.path.join(
            args.index_path,
            TERM_ID_SUBPATH
        ),
        "rb"
    ) as f:
        term_to_id = pickle.load(f)

    with open(
        args.queries_file,
        "r",
        encoding="utf-8"
    ) as f:

        queries = [
            line.strip()
            for line in f
            if line.strip()
        ]

    by_query_length = {}
    by_posting_size = {}

    global_taat = []
    global_daat = []

    print(
        f"Procesando {len(queries)} queries..."
    )

    for query in queries:

        taat_time, daat_time = benchmark_query(
            args.index_path,
            query,
            args.k
        )

        global_taat.append(taat_time)
        global_daat.append(daat_time)

        q_bucket = query_length_bucket(
            query
        )

        avg_posting = average_posting_size(
            query,
            vocabulary,
            term_to_id
        )

        p_bucket = posting_bucket(
            avg_posting
        )

        by_query_length.setdefault(
            q_bucket,
            {
                "taat": [],
                "daat": []
            }
        )

        by_posting_size.setdefault(
            p_bucket,
            {
                "taat": [],
                "daat": []
            }
        )

        by_query_length[q_bucket]["taat"].append(
            taat_time
        )

        by_query_length[q_bucket]["daat"].append(
            daat_time
        )

        by_posting_size[p_bucket]["taat"].append(
            taat_time
        )

        by_posting_size[p_bucket]["daat"].append(
            daat_time
        )

        print(
            f"{query} | "
            f"TAAT={taat_time:.6f}s | "
            f"DAAT={daat_time:.6f}s"
        )

    print()
    print("=" * 60)
    print("RESULTADOS GLOBALES")
    print("=" * 60)

    print(
        f"TAAT promedio : "
        f"{statistics.mean(global_taat):.6f} s"
    )

    print(
        f"DAAT promedio : "
        f"{statistics.mean(global_daat):.6f} s"
    )

    print_group_results(
        "POR LONGITUD DE QUERY",
        by_query_length
    )

    print_group_results(
        "POR TAMAÑO DE POSTING LIST",
        by_posting_size
    )


if __name__ == "__main__":
    main()