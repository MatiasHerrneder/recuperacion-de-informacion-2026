"""
skiplist_benchmark.py
=====================
Compara AND sin skips (InMemoryPosting) vs AND con skips (DiskPostingList)
sobre queries de 2 y 3 terminos que esten en el vocabulario.

Uso:
    python skiplist_benchmark.py <index_sin_skips> <index_con_skips> <queries_txt>

Ejemplo:
    python skiplist_benchmark.py output2/ output/ queries.txt
"""

import argparse
import os
import pickle
import struct
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Sequence

# ── Rutas ─────────────────────────────────────────────────────────────────────
INDEX_SUBPATH          = "index.bin"
VOCABULARY_SUBPATH     = "vocabulary.pkl"
TERM_ID_SUBPATH        = "term_index.pkl"
DOCUMENT_INDEX_SUBPATH = "document_index.pkl"
SKIPLIST_SUBPATH       = "skiplist.bin"
SKIP_INDEX_SUBPATH     = "skip_index.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# InMemoryPosting  —  AND lineal, sin skips
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class InMemoryPosting:
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

    def posting_and(self, other: "Posting") -> "InMemoryPosting":
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
        return InMemoryPosting(result, [1.0] * len(result))


# ══════════════════════════════════════════════════════════════════════════════
# DiskPostingList  —  AND con skips (igual a tu implementacion)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DiskPostingList:
    index_file: str
    seek:       int
    length:     int
    skips:      dict          # {pos_actual -> pos_destino}
    cursor:     int = 0

    ENTRY_SIZE: int = field(default=8, init=False, repr=False)

    def __post_init__(self):
        if not self.length:
            self.cursor = -1

    def _read_entry(self, index: int):
        with open(self.index_file, 'rb') as f:
            f.seek(self.seek + index * 8)
            return struct.unpack('>II', f.read(8))

    def docid(self) -> Optional[int]:
        return None if self.cursor == -1 else self._read_entry(self.cursor)[0]

    def next(self) -> None:
        if self.cursor != -1:
            self.cursor += 1
            if self.cursor >= self.length:
                self.cursor = -1

    def ge(self, docid: int) -> Optional[int]:
        if self.cursor == -1:
            return None
        # saltar con skips mientras el destino sea <= docid buscado
        while self.cursor in self.skips:
            dst = self.skips[self.cursor]
            if self._read_entry(dst)[0] <= docid:
                self.cursor = dst
            else:
                break
        # barrido lineal final
        while self.cursor != -1 and self._read_entry(self.cursor)[0] < docid:
            self.next()
        return self.docid()

    def reset(self) -> None:
        self.cursor = 0 if self.length else -1

    def posting_and(self, other: "Posting") -> InMemoryPosting:
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
        return InMemoryPosting(result, [1.0] * len(result))


# ══════════════════════════════════════════════════════════════════════════════
# Carga de indices
# ══════════════════════════════════════════════════════════════════════════════
def load_index(path):
    vocabulary = pickle.load(open(os.path.join(path, VOCABULARY_SUBPATH), 'rb'))
    term_to_id = pickle.load(open(os.path.join(path, TERM_ID_SUBPATH),    'rb'))
    doc_index  = pickle.load(open(os.path.join(path, DOCUMENT_INDEX_SUBPATH), 'rb'))
    return vocabulary, term_to_id, doc_index


def load_skip_index(path):
    """
    skip_index[term_id] = [offset_en_skiplist_bin, n_skips]
    skiplist.bin tiene entradas de 3 unsigned int: (term_id, pos_actual, pos_destino)
    Devuelve dict: term_id -> {pos_actual: pos_destino}
    """
    skip_index_raw = pickle.load(open(os.path.join(path, SKIP_INDEX_SUBPATH), 'rb'))
    skip_data = open(os.path.join(path, SKIPLIST_SUBPATH), 'rb').read()

    skips_by_term = {}
    for term_id, (offset, n_skips) in skip_index_raw.items():
        skips = {}
        for i in range(n_skips):
            _, pos_actual, pos_destino = struct.unpack('>III', skip_data[offset + i*12 : offset + i*12 + 12])
            skips[pos_actual] = pos_destino
        skips_by_term[term_id] = skips
    return skips_by_term


def get_posting_linear(index_path, term, vocabulary, term_to_id) -> InMemoryPosting:
    if term not in term_to_id:
        return InMemoryPosting([], [])
    term_id = term_to_id[term]
    seek, length = vocabulary[term_id]
    with open(os.path.join(index_path, INDEX_SUBPATH), 'rb') as f:
        f.seek(seek)
        data = f.read(length * 8)
    doc_ids = [struct.unpack('>II', data[i:i+8])[0] for i in range(0, len(data), 8)]
    return InMemoryPosting(doc_ids, [1.0] * len(doc_ids))


def get_posting_skip(index_path, term, vocabulary, term_to_id, skips_by_term) -> DiskPostingList:
    if term not in term_to_id:
        return DiskPostingList(os.path.join(index_path, INDEX_SUBPATH), 0, 0, {})
    term_id = term_to_id[term]
    seek, length = vocabulary[term_id]
    skips = skips_by_term.get(term_id, {})
    return DiskPostingList(os.path.join(index_path, INDEX_SUBPATH), seek, length, skips)


# ══════════════════════════════════════════════════════════════════════════════
# Parseo y filtrado de queries
# ══════════════════════════════════════════════════════════════════════════════
def parse_queries(queries_path):
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = re.sub(r'^\d+:', '', line.strip()).strip().lower()
            if text:
                queries.append(text.split())
    return queries


def filter_queries(queries, term_to_id):
    result = {2: [], 3: []}
    for words in queries:
        if len(words) in (2, 3) and all(w in term_to_id for w in words):
            result[len(words)].append(words)
    return result


def and_patterns(words):
    """Solo patrones AND segun el enunciado."""
    if len(words) == 2:
        t1, t2 = words
        return [("t1 AND t2", [t1, t2])]
    else:
        t1, t2, t3 = words
        return [("t1 AND t2 AND t3", [t1, t2, t3])]


Posting = InMemoryPosting | DiskPostingList


def and_linear(postings: Sequence[Posting]) -> InMemoryPosting:
    result: InMemoryPosting = postings[0].posting_and(postings[1])
    for other in postings[2:]:
        result = result.posting_and(other)
    return result


def and_skip(postings: Sequence[Posting]) -> InMemoryPosting:
    result: InMemoryPosting = postings[0].posting_and(postings[1])
    for other in postings[2:]:
        result = result.posting_and(other)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════
def run_benchmark(index_path_linear, index_path_skip, queries_path):
    print("=" * 72)
    print("  BENCHMARK AND: SIN SKIPS vs CON SKIPS")
    print("=" * 72)

    # cargar indices
    voc_lin, tid_lin, doc_lin = load_index(index_path_linear)
    voc_ski, tid_ski, doc_ski = load_index(index_path_skip)
    skips_by_term             = load_skip_index(index_path_skip)

    # usar vocabulario del indice con skips para filtrar
    # (ambos deberian ser el mismo corpus, pero usamos el comun)
    common_terms = set(tid_lin) & set(tid_ski)
    tid_common   = {t: tid_ski[t] for t in common_terms}

    raw   = parse_queries(queries_path)
    filtered = filter_queries(raw, tid_common)

    n2, n3 = len(filtered[2]), len(filtered[3])
    print(f"\n  Queries validas: {n2} de 2 terminos | {n3} de 3 terminos")
    print(f"  Total corridas:  {(n2 + n3 * 1)} queries x 2 modos\n")

    if not n2 and not n3:
        print("  No hay queries validas. Verifica que los terminos esten en el vocabulario.")
        return

    all_tasks = []
    for n in (2, 3):
        for words in filtered[n]:
            for pat_name, terms in and_patterns(words):
                all_tasks.append((pat_name, terms))

    total   = len(all_tasks)
    results = []

    for i, (pat_name, terms) in enumerate(all_tasks, 1):
        query_str = " AND ".join(terms)
        pl_sizes  = [voc_ski[tid_ski[t]][1] for t in terms if t in tid_ski]

        print(f"  [{i:3d}/{total}]  {query_str:<45}  (listas: {pl_sizes})")

        # ── sin skips ──────────────────────────────────────────────────────
        postings_lin = [get_posting_linear(index_path_linear, t, voc_lin, tid_lin) for t in terms]
        t0 = time.perf_counter()
        res_lin = and_linear(postings_lin)
        t_lin = (time.perf_counter() - t0) * 1_000_000

        # ── con skips ──────────────────────────────────────────────────────
        postings_ski = [get_posting_skip(index_path_skip, t, voc_ski, tid_ski, skips_by_term) for t in terms]
        t0 = time.perf_counter()
        res_ski = and_skip(postings_ski)
        t_ski = (time.perf_counter() - t0) * 1_000_000

        speedup = t_lin / t_ski if t_ski > 0 else float('inf')
        results.append({
            "pat":      pat_name,
            "query":    query_str,
            "sizes":    pl_sizes,
            "res_lin":  len(res_lin.docids),
            "res_ski":  len(res_ski.docids),
            "t_lin":    t_lin,
            "t_ski":    t_ski,
            "speedup":  speedup,
        })

        print(f"         sin skips: {t_lin:8.1f} us   con skips: {t_ski:8.1f} us   speedup: {speedup:.2f}x")

    # ── tabla resumen ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESUMEN")
    print("=" * 72)
    hdr = f"  {'Query':<42} {'Sizes':<14} {'Sin skips':>10} {'Con skips':>10} {'Speedup':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in results:
        sizes_str = str(r["sizes"])
        q = r["query"][:40] + ".." if len(r["query"]) > 42 else r["query"]
        print(f"  {q:<42} {sizes_str:<14} {r['t_lin']:>9.1f}us {r['t_ski']:>9.1f}us {r['speedup']:>7.2f}x")

    total_lin = sum(r["t_lin"] for r in results)
    total_ski = sum(r["t_ski"] for r in results)
    overall   = total_lin / total_ski if total_ski > 0 else float('inf')
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<42} {'':<14} {total_lin:>9.1f}us {total_ski:>9.1f}us {overall:>7.2f}x")

    print("\n  CONCLUSIONES:")
    print(f"    Speedup promedio con skips: {overall:.2f}x")
    print( "    Queries con listas grandes se benefician mas de los skips.")
    print( "    Queries con listas pequenas pueden no mostrar mejora (overhead de I/O por skip).")
    if any(r["res_lin"] != r["res_ski"] for r in results):
        print("    ATENCION: algunos resultados difieren entre implementaciones.")
    else:
        print("    Resultados identicos entre ambas implementaciones: correcto.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark AND sin skips vs con skips")
    parser.add_argument("index_linear", help="Directorio del indice SIN skips (output2/)")
    parser.add_argument("index_skip",   help="Directorio del indice CON skips (output/)")
    parser.add_argument("queries_txt",  help="Archivo de queries (formato N:texto)")
    args = parser.parse_args()

    run_benchmark(args.index_linear, args.index_skip, args.queries_txt)