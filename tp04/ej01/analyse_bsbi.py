import argparse
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from indexer import index_bsbi


# ─────────────────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def count_files(corpus_path: str) -> int:
    """Cuenta archivos en el corpus (recursivamente)."""
    total = 0
    for _, _, files in os.walk(corpus_path):
        total += sum(1 for f in files if f.lower().endswith(".html"))
    return total


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# grafico tiempos de indexacion vs merge para distintos n

def plot_times(ns: list[int], idx_times: list[float], merge_times: list[float],
               save_path: str):
    x     = np.arange(len(ns))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#f7f7f7")
    fig.patch.set_facecolor("white")

    bars_idx   = ax.bar(x - width/2, idx_times,   width, label="Indexación",
                        color="#4C72B0", alpha=0.88, zorder=3)
    bars_merge = ax.bar(x + width/2, merge_times, width, label="Merge",
                        color="#DD8452", alpha=0.88, zorder=3)

    ax.set_xlabel("n  (docs por volcado)", fontsize=11)
    ax.set_ylabel("Tiempo (s)", fontsize=11)
    ax.set_title("Tiempo de indexacion vs merge por valor de n", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in ns])
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # etiquetas encima de cada barra
    for bar in bars_idx:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.2f}s", ha="center", va="bottom", fontsize=8, color="#333")
    for bar in bars_merge:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.2f}s", ha="center", va="bottom", fontsize=8, color="#333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# grafico distribucion de tamaños de posting lists (histograma log-log)


def plot_posting_distribution(lengths: list[int], save_path: str, n_label: str):
    if not lengths:
        print("No hay datos de posting lists para graficar.")
        return

    arr = np.array(lengths, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Distribución de tamaños de posting lists  (n={n_label})",
        fontsize=13, fontweight="bold"
    )

    # subplot izquierdo: histograma en escala logaritmica
    ax = axes[0]
    ax.set_facecolor("#f7f7f7")
    max_len  = int(arr.max())
    bins_log = np.logspace(0, math.log10(max_len + 1), 60)
    ax.hist(arr, bins=bins_log, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Longitud de posting list (# docs)", fontsize=10)
    ax.set_ylabel("Cantidad de términos (frecuencia)", fontsize=10)
    ax.set_title("Histograma log-log", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # subplot derecho: CDF
    ax2 = axes[1]
    ax2.set_facecolor("#f7f7f7")
    sorted_arr = np.sort(arr)
    cdf        = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    ax2.plot(sorted_arr, cdf, color="#DD8452", linewidth=1.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("Longitud de posting list (# docs)", fontsize=10)
    ax2.set_ylabel("Proporción acumulada de términos", fontsize=10)
    ax2.set_title("Distribución acumulada (CDF)", fontsize=11)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)

    # estadisticas en el margen
    stats_text = (
        f"términos: {len(arr):,}\n"
        f"mín: {int(arr.min())}\n"
        f"mediana: {int(np.median(arr))}\n"
        f"media: {arr.mean():.1f}\n"
        f"máx: {int(arr.max()):,}"
    )
    fig.text(0.99, 0.5, stats_text, ha="right", va="center",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#eeeeee", alpha=0.7))

    plt.tight_layout(rect=(0, 0, 0.88, 1))
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# grafico overhead del índice

def plot_overhead(ns: list[int], ratios: list[float], corpus_mb: float,
                  index_mbs: list[float], save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Overhead del índice respecto al corpus", fontsize=13, fontweight="bold")

    # ratio
    ax1.set_facecolor("#f7f7f7")
    ax1.plot([str(n) for n in ns], [r * 100 for r in ratios],
             marker="o", linewidth=2, color="#4C72B0", markersize=7, zorder=3)
    for x_val, y_val in zip([str(n) for n in ns], [r * 100 for r in ratios]):
        ax1.annotate(f"{y_val:.1f}%", (x_val, y_val),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#333")
    ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="100 % = tamaño corpus")
    ax1.set_xlabel("n  (docs por volcado)", fontsize=10)
    ax1.set_ylabel("Overhead  (%)", fontsize=10)
    ax1.set_title("Índice / Corpus  (%)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    # tamaños absolutos
    ax2.set_facecolor("#f7f7f7")
    x_pos = np.arange(len(ns))
    ax2.bar(x_pos, index_mbs, color="#55A868", alpha=0.85, zorder=3,
            label=f"Índice (MB)")
    ax2.axhline(y=corpus_mb, color="#C44E52", linestyle="--", linewidth=1.8,
                label=f"Corpus: {corpus_mb:.1f} MB")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel("n  (docs por volcado)", fontsize=10)
    ax2.set_ylabel("Tamaño (MB)", fontsize=10)
    ax2.set_title("Tamaño absoluto del índice vs corpus", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de BSBI con distintos valores de n."
    )
    parser.add_argument("corpus_path", help="Directorio raíz del corpus")
    parser.add_argument("--stop_words_path", default=None)
    parser.add_argument("--output_base",     default="bsbi_experiments",
                        help="Directorio base donde guardar resultados")
    parser.add_argument("--ns", type=int, nargs="+",
                        help="Valores de n a probar. Si se omite, se calculan como "
                             "5 %%, 10 %%, 25 %%, 50 %%, 100 %% del total de docs.")
    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)

    total_docs = count_files(args.corpus_path)
    if total_docs == 0:
        print("No se encontraron archivos validos en el corpus.")
        return

    print(f"\nCorpus: {args.corpus_path}")
    print(f"Total de documentos: {total_docs:,}")

    if args.ns:
        ns = sorted(set(args.ns))
    else:
        # porcentajes representativos del tamaño de la colección
        percentages = [0.05, 0.10, 0.25, 0.50]
        ns = sorted(set(max(1, int(p * total_docs)) for p in percentages))

    print(f"Valores de n a probar: {ns}\n")

    # metricas
    idx_times = []
    merge_times = []
    ratios = []
    index_mbs = []
    corpus_mb: float = 0.0
    all_lengths = {} # n -> posting list lengths

    for n in ns:
        out_dir = os.path.join(args.output_base, f"n_{n}")
        print(f"{'─'*50}")
        print(f"  Ejecutando con n = {n} …")

        m = index_bsbi(
            corpus_path     = args.corpus_path,
            memory_limit    = n,
            output_dir      = out_dir,
            stop_words_path = args.stop_words_path,
        )

        idx_times.append(m["indexing_time"])
        merge_times.append(m["merge_time"])
        ratios.append(m["overhead_ratio"])
        index_mbs.append(m["index_size_bytes"] / 1_048_576)
        all_lengths[n] = m["posting_list_lengths"]

        if corpus_mb is None:
            corpus_mb = m["corpus_size_bytes"] / 1_048_576

        print(f"  Indexación : {m['indexing_time']:.3f} s")
        print(f"  Merge      : {m['merge_time']:.3f} s")
        print(f"  Chunks     : {m['num_chunks']}")
        print(f"  Términos   : {m['num_terms']:,}")
        print(f"  Docs       : {m['num_docs']:,}")
        print(f"  Corpus     : {human_bytes(m['corpus_size_bytes'])}")
        print(f"  Índice     : {human_bytes(m['index_size_bytes'])}")
        print(f"  Overhead   : {m['overhead_ratio']*100:.2f}%")

    # graficos
    
    graphs_dir = os.path.join(args.output_base, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    print(f"\n{'─'*50}")
    print("Generando gráficas…")

    # tiempos
    plot_times(
        ns, idx_times, merge_times,
        save_path=os.path.join(graphs_dir, "tiempos.png")
    )

    # distribucion de posting lists
    n_ref = ns[1] if len(ns) > 1 else ns[0]
    plot_posting_distribution(
        all_lengths[n_ref],
        save_path=os.path.join(graphs_dir, "posting_distribution.png"),
        n_label=str(n_ref)
    )

    # overhead
    plot_overhead(
        ns, ratios, corpus_mb, index_mbs,
        save_path=os.path.join(graphs_dir, "overhead.png")
    )

    # conclusiones
    print(f"\n{'═'*50}")
    print("  CONCLUSIONES")
    print(f"{'═'*50}")

    best_n_idx = int(np.argmin([i + m for i, m in zip(idx_times, merge_times)]))
    worst_n_idx = int(np.argmax([i + m for i, m in zip(idx_times, merge_times)]))

    print(f"\n  • n óptimo (menor tiempo total): n = {ns[best_n_idx]}"
          f"  ({(idx_times[best_n_idx]+merge_times[best_n_idx]):.2f} s)")
    print(f"  • n más lento                  : n = {ns[worst_n_idx]}"
          f"  ({(idx_times[worst_n_idx]+merge_times[worst_n_idx]):.2f} s)")

    max_ratio = max(ratios)
    min_ratio = min(ratios)
    print(f"\n  • Overhead máximo: {max_ratio*100:.2f}%  (n={ns[ratios.index(max_ratio)]})")
    print(f"  • Overhead mínimo: {min_ratio*100:.2f}%  (n={ns[ratios.index(min_ratio)]})")
    print(f"  • El overhead es {'estable' if (max_ratio-min_ratio)<0.01 else 'variable'}"
          " entre valores de n → el tamaño del índice no depende fuertemente de n.")

    # estadisticas de posting lists del n de referencia
    arr = np.array(all_lengths[n_ref])
    p90 = np.percentile(arr, 90)
    print(f"\n  • Posting lists (n={n_ref}): "
          f"mediana={int(np.median(arr))}, media={arr.mean():.1f}, "
          f"máx={int(arr.max()):,}")
    print(f"    El 90 % de los términos tiene posting list ≤ {int(p90)} docs "
          f"→ distribución muy sesgada a la derecha (ley de Zipf).")

    print(f"\n  Gráficas en: {graphs_dir}/\n")


if __name__ == "__main__":
    main()