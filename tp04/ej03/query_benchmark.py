"""
boolean_query_benchmark.py
==========================
Ejecuta consultas booleanas sobre el índice BSBI y mide tiempos de ejecución.
Compara modo DISCO (posting leído en cada consulta) vs MEMORIA (índice precargado).

Uso:
    python boolean_query_benchmark.py <index_path> <queries_txt>

Ejemplo:
    python boolean_query_benchmark.py output2/ queries.txt
"""

import argparse
import os
import pickle
import struct
import re
import time
from dataclasses import dataclass
from typing import Optional, List

# ── Rutas de archivos del índice ─────────────────────────────────────────────
INDEX_SUBPATH      = "index.bin"
VOCABULARY_SUBPATH = "vocabulary.pkl"
TERM_ID_SUBPATH    = "term_index.pkl"
DOCUMENT_INDEX_SUBPATH = "document_index.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# PostingList en memoria (igual a tu InMemoryPosting)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PostingList:
    docids:  List[int]
    weights: List[float]
    cursor:  int = 0

    def __post_init__(self):
        if not self.docids:
            self.cursor = -1

    def docid(self) -> Optional[int]:
        return None if self.cursor == -1 else self.docids[self.cursor]

    def next(self) -> None:
        if self.cursor != -1:
            self.cursor += 1
            if self.cursor >= len(self.docids):
                self.cursor = -1

    def ge(self, docid: int) -> Optional[int]:
        while self.cursor != -1 and self.docids[self.cursor] < docid:
            self.next()
        return self.docid()

    def reset(self) -> None:
        self.cursor = 0 if self.docids else -1

    def posting_and(self, other: "PostingList") -> "PostingList":
        self.reset(); other.reset()
        result = []
        a, b = self.docid(), other.docid()
        while a is not None and b is not None:
            if a == b:
                result.append(a); self.next(); other.next()
            elif a < b:
                self.ge(b)
            else:
                other.ge(a)
            a, b = self.docid(), other.docid()
        return PostingList(result, [1.0] * len(result))

    def posting_or(self, other: "PostingList") -> "PostingList":
        self.reset(); other.reset()
        result = []
        a, b = self.docid(), other.docid()
        while a is not None or b is not None:
            if a is not None and (b is None or a < b):
                result.append(a); self.next()
            elif b is not None and (a is None or b < a):
                result.append(b); other.next()
            else:
                result.append(a); self.next(); other.next()
            a, b = self.docid(), other.docid()
        return PostingList(result, [1.0] * len(result))

    def posting_not(self, universe: "PostingList") -> "PostingList":
        self.reset(); universe.reset()
        result = []
        while (u := universe.docid()) is not None:
            if self.ge(u) != u:
                result.append(u)
            universe.next()
        return PostingList(result, [1.0] * len(result))


# ══════════════════════════════════════════════════════════════════════════════
# Recuperación de postings
# ══════════════════════════════════════════════════════════════════════════════
def retrieve_posting_disk(index_path, term, vocabulary, term_to_id) -> PostingList:
    """Lee la posting list del disco en cada llamada."""
    if term not in term_to_id:
        return PostingList([], [])
    term_id = term_to_id[term]
    seek, length = vocabulary[term_id]
    with open(os.path.join(index_path, INDEX_SUBPATH), 'rb') as f:
        f.seek(seek)
        data = f.read(length * 8)
    doc_ids, freqs = [], []
    for i in range(0, len(data), 8):
        doc_id, freq = struct.unpack('>2I', data[i:i+8])
        doc_ids.append(doc_id); freqs.append(freq)
    return PostingList(doc_ids, [float(w) for w in freqs])


def retrieve_posting_memory(term, vocabulary, term_to_id, full_index_bytes) -> PostingList:
    """Lee la posting list desde el índice ya cargado en memoria."""
    if term not in term_to_id:
        return PostingList([], [])
    term_id = term_to_id[term]
    seek, length = vocabulary[term_id]
    data = full_index_bytes[seek: seek + length * 8]
    doc_ids, freqs = [], []
    for i in range(0, len(data), 8):
        doc_id, freq = struct.unpack('>2I', data[i:i+8])
        doc_ids.append(doc_id); freqs.append(freq)
    return PostingList(doc_ids, [float(w) for w in freqs])


# ══════════════════════════════════════════════════════════════════════════════
# Evaluador booleano TAAT (Shunting-Yard + stack)
# ══════════════════════════════════════════════════════════════════════════════
PRECEDENCE = {'NOT': 3, 'AND': 2, 'OR': 1}

def tokenize(query: str) -> list:
    return re.findall(r'\(|\)|AND|OR|NOT|\w+', query)

def to_postfix(tokens: list) -> list:
    output, operators = [], []
    for token in tokens:
        if token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
        elif token in PRECEDENCE:
            while (operators and operators[-1] != '('
                   and operators[-1] in PRECEDENCE
                   and PRECEDENCE[operators[-1]] >= PRECEDENCE[token]):
                output.append(operators.pop())
            operators.append(token)
        else:
            output.append(token)
    while operators:
        output.append(operators.pop())
    return output

def evaluate(postfix, get_posting_fn, universe: PostingList) -> PostingList:
    stack = []
    for token in postfix:
        if token == 'NOT':
            a = stack.pop()
            stack.append(a.posting_not(universe))
        elif token == 'AND':
            b, a = stack.pop(), stack.pop()
            stack.append(a.posting_and(b))
        elif token == 'OR':
            b, a = stack.pop(), stack.pop()
            stack.append(a.posting_or(b))
        else:
            stack.append(get_posting_fn(token))
    return stack.pop() if stack else PostingList([], [])


# ══════════════════════════════════════════════════════════════════════════════
# Generador de patrones booleanos
# ══════════════════════════════════════════════════════════════════════════════
def make_patterns(terms: List[str]) -> List[tuple]:
    """
    Devuelve lista de (nombre_patron, query_string) según el enunciado.
    - 2 términos: AND, OR, NOT
    - 3 términos: AND AND, (OR) NOT, (AND) OR
    """
    if len(terms) == 2:
        t1, t2 = terms
        return [
            ("t1 AND t2",  f"{t1} AND {t2}"),
            ("t1 OR t2",   f"{t1} OR {t2}"),
            ("t1 NOT t2",  f"{t1} NOT {t2}"),
        ]
    elif len(terms) == 3:
        t1, t2, t3 = terms
        return [
            ("t1 AND t2 AND t3",   f"{t1} AND {t2} AND {t3}"),
            ("(t1 OR t2) NOT t3",  f"({t1} OR {t2}) NOT {t3}"),
            ("(t1 AND t2) OR t3",  f"({t1} AND {t2}) OR {t3}"),
        ]
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Carga del índice
# ══════════════════════════════════════════════════════════════════════════════
def load_index_files(index_path):
    with open(os.path.join(index_path, VOCABULARY_SUBPATH), 'rb') as f:
        vocabulary = pickle.load(f)
    with open(os.path.join(index_path, TERM_ID_SUBPATH), 'rb') as f:
        term_to_id = pickle.load(f)
    with open(os.path.join(index_path, DOCUMENT_INDEX_SUBPATH), 'rb') as f:
        doc_index = pickle.load(f)
    return vocabulary, term_to_id, doc_index


def load_full_index_bytes(index_path) -> bytes:
    with open(os.path.join(index_path, INDEX_SUBPATH), 'rb') as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════════════
# Parseo del archivo de queries
# ══════════════════════════════════════════════════════════════════════════════
def parse_queries(queries_path: str) -> List[List[str]]:
    """
    Lee el archivo de queries con formato  N:texto de la query
    Devuelve lista de listas de palabras (ya en minúsculas).
    """
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # eliminar prefijo numérico  "1:", "12:", etc.
            text = re.sub(r'^\d+:', '', line).strip().lower()
            words = text.split()
            queries.append(words)
    return queries


def filter_queries(queries: List[List[str]], term_to_id: dict) -> dict:
    """
    Filtra queries de 2 y 3 términos cuyos términos estén TODOS en el vocabulario.
    Devuelve dict: {2: [...], 3: [...]}
    """
    result = {2: [], 3: []}
    for words in queries:
        n = len(words)
        if n not in (2, 3):
            continue
        if all(w in term_to_id for w in words):
            result[n].append(words)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Ejecución del benchmark
# ══════════════════════════════════════════════════════════════════════════════
def run_benchmark(index_path, queries_path):
    print("=" * 70)
    print("  BENCHMARK DE CONSULTAS BOOLEANAS")
    print("=" * 70)

    # Cargar estructuras auxiliares
    vocabulary, term_to_id, doc_index = load_index_files(index_path)
    universe = PostingList(sorted(doc_index.keys()), [1.0] * len(doc_index))
    num_docs = len(doc_index)

    print(f"\n  Índice cargado:  {len(term_to_id):,} términos | {num_docs:,} documentos")
    print(f"  Vocabulario:     {len(vocabulary):,} entradas")

    # Leer y filtrar queries
    raw_queries = parse_queries(queries_path)
    filtered = filter_queries(raw_queries, term_to_id)

    print(f"\n  Queries válidas: {len(filtered[2])} de 2 términos | {len(filtered[3])} de 3 términos")

    if not filtered[2] and not filtered[3]:
        print("\n  ⚠ Ninguna query pasó el filtro. Verificá que los términos estén en el vocabulario.")
        return

    # ── MODO DISCO ────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  MODO: DISCO  (posting leída en cada consulta)")
    print("─" * 70)

    results_disk = run_mode(
        filtered, index_path, vocabulary, term_to_id, universe,
        mode="disk", full_index_bytes=None
    )

    # ── MODO MEMORIA ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  MODO: MEMORIA  (índice completo precargado en RAM)")
    print("─" * 70)

    t_load_start = time.perf_counter()
    full_index_bytes = load_full_index_bytes(index_path)
    t_load_end = time.perf_counter()
    print(f"\n  Tiempo de carga en memoria: {(t_load_end - t_load_start)*1000:.2f} ms  "
          f"({len(full_index_bytes)/1024:.1f} KB)")

    results_mem = run_mode(
        filtered, index_path, vocabulary, term_to_id, universe,
        mode="memory", full_index_bytes=full_index_bytes
    )

    # ── COMPARATIVA ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARATIVA DISCO vs MEMORIA")
    print("=" * 70)
    print_comparison(results_disk, results_mem, vocabulary, term_to_id)


def run_mode(filtered, index_path, vocabulary, term_to_id, universe,
             mode, full_index_bytes):
    results = {}

    if mode == 'disk':
        get_posting = lambda term: retrieve_posting_disk(index_path, term, vocabulary, term_to_id)
    else:
        get_posting = lambda term: retrieve_posting_memory(term, vocabulary, term_to_id, full_index_bytes)

    all_tasks = []
    for n_terms in (2, 3):
        for words in filtered[n_terms]:
            for pat_name, query_str in make_patterns(words):
                all_tasks.append((n_terms, words, pat_name, query_str))

    total = len(all_tasks)

    for i, (n_terms, words, pat_name, query_str) in enumerate(all_tasks, 1):
        print(f"[{mode:6s}] {i}/{total}  {' '.join(words):<30}  {pat_name}", flush=True)

        postfix = to_postfix(tokenize(query_str))

        t_start = time.perf_counter()
        result_pl = evaluate(postfix, get_posting, universe)
        t_end = time.perf_counter()

        elapsed_us = (t_end - t_start) * 1_000_000
        pl_sizes = [vocabulary[term_to_id[w]][1] for w in words if w in term_to_id]

        results[(n_terms, pat_name, query_str)] = {
            "elapsed_us": elapsed_us,
            "result_size": len(result_pl.docids),
            "pl_sizes": pl_sizes,
            "words": words,
        }

    return results


def print_comparison(results_disk, results_mem, vocabulary, term_to_id):
    header = f"  {'Patrón':<22} {'Query':<40} {'Disco (µs)':>12} {'Memoria (µs)':>13} {'Speedup':>9}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for key in results_disk:
        td = results_disk[key]["elapsed_us"]
        tm = results_mem[key]["elapsed_us"]
        speedup = td / tm if tm > 0 else float('inf')
        _, pat_name, query_str = key
        # Truncar query si es muy larga
        q_short = query_str[:38] + ".." if len(query_str) > 40 else query_str
        print(f"  {pat_name:<22} {q_short:<40} {td:>12.1f} {tm:>13.1f} {speedup:>8.2f}x")

    # Totales
    total_disk = sum(v["elapsed_us"] for v in results_disk.values())
    total_mem  = sum(v["elapsed_us"] for v in results_mem.values())
    overall_speedup = total_disk / total_mem if total_mem > 0 else float('inf')
    print("  " + "─" * (len(header) - 2))
    print(f"  {'TOTAL':<22} {'':<40} {total_disk:>12.1f} {total_mem:>13.1f} {overall_speedup:>8.2f}x")

    print("\n  CONCLUSIONES:")
    print(f"    • Speedup promedio memoria vs disco: {overall_speedup:.2f}x")
    if overall_speedup > 1.5:
        print("    • Acceso a disco introduce latencia significativa de I/O.")
    else:
        print("    • El sistema de archivos/SO cachea bien el índice; diferencia acotada.")
    print("    • Queries con listas grandes (OR) son más lentas que con listas pequeñas (AND).")
    print("    • NOT sobre el universo completo puede ser muy costoso si el complemento es grande.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark de consultas booleanas sobre índice BSBI"
    )
    parser.add_argument("index_path",  help="Directorio con los archivos del índice (output2/)")
    parser.add_argument("queries_txt", help="Archivo de queries (formato N:texto)")
    args = parser.parse_args()

    run_benchmark(args.index_path, args.queries_txt)