from stack import Queue
from extract_links import request_and_extract_links
from urllib.parse import urlparse
import pickle
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def crawl(initial_urls: list, max_page_per_domain: int, max_logical_depth: int, max_physical_depth: int, max_total_pages: int):
    todo_queue = Queue() # todo_list
    visited_urls = set() # done_list
    graph = {} # replaces push()
    url_ids = {}
    id_count = 0
    logical_depth = {}
    pages_per_domain = {}

    # restricts to initial domains
    seed_domains = {get_domain(url) for url in initial_urls}

    for url in initial_urls:
        todo_queue.enqueue(url)
        url_ids[url] = id_count
        logical_depth[url] = 0
        pages_per_domain[get_domain(url)] = pages_per_domain.get(get_domain(url), 0) + 1
        id_count += 1

    while not todo_queue.is_empty():

        if len(visited_urls) >= max_total_pages:
            print(f"Límite de {max_total_pages} páginas alcanzado.")
            break

        print(todo_queue.size())
        curr_url = todo_queue.peek()
        todo_queue.dequeue()

        visited_urls.add(curr_url)
        graph[curr_url] = []

        current_domain = get_domain(curr_url)
        current_logical_depth = logical_depth.get(curr_url, 0)

        try:
            links = request_and_extract_links(curr_url)
        except Exception as e:
            print(f"Error fetching {curr_url}: {e}")
            continue
        
        for link in links:
            if link in visited_urls or link in url_ids:
                if link in url_ids:
                    graph[curr_url].append(url_ids[link])

            else:
                if len(url_ids) >= max_total_pages:
                    continue

                link_domain = get_domain(link)
                link_logical_depth = current_logical_depth + 1
                
                if (passes_url_filter(link) and 
                pages_per_domain.get(get_domain(link), 0) < max_page_per_domain and 
                link_logical_depth <= max_logical_depth and 
                physical_depth(link) <= max_physical_depth):
                # physical_depth(link) <= max_physical_depth and
                # urlparse(link).netloc in seed_domains):
                
                    logical_depth[link] = link_logical_depth
                    todo_queue.enqueue(link)
                    url_ids[link] = id_count
                    id_count += 1
                    pages_per_domain[link_domain] = pages_per_domain.get(link_domain, 0) + 1
                    graph[curr_url].append(url_ids[link])

    url_ids = {url: uid for url, uid in url_ids.items() if url in visited_urls}
    return graph, url_ids


def get_domain(url):
    return urlparse(url).netloc

def physical_depth(url):
    path = urlparse(url).path
    segments = [s for s in path.split("/") if s]
    return len(segments)

def passes_url_filter(url: str)-> bool:
    '''
    Defines whether a parameter is valid to crawl
    Args:
        url (str):
    Returns:
        bool:
    '''

    return urlparse(url).scheme in ("http", "https")


def save_graph(graph, url_ids, filename="graph.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"graph": graph, "url_ids": url_ids}, f)

def load_graph(filename="graph.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["graph"], data["url_ids"]

def visualize_graph(graph, url_ids, filename="graph.html"):
    id_to_url = {v: k for k, v in url_ids.items()}
    
    net = Network(height="750px", width="100%", directed=True)
    net.barnes_hut()

    # agregar nodos
    for url, node_id in url_ids.items():
        label = f"{urlparse(url).netloc}{urlparse(url).path}"
        net.add_node(node_id, label=label, title=url)

    # agregar edges
    for src_url, outlinks in graph.items():
        src_id = url_ids[src_url]
        for dst_id in outlinks:
            net.add_edge(src_id, dst_id)

    net.show(filename, notebook=False)


# push(todo_list, initial_set_of_urls)
# while todo_list[0] != {}:
#     page = fetch_page(todo_list[0])
#     if page downloaded:
#         links = parse(page)
#         for link in links:
#             if link in done_list:
#                 push(todo_list[0].outlinks, done_list[link].id)
#             elif link in todo_list:
#                 push(todo_list[0].outlinks, todo_list[link].id)
#             elif link passourfilter:
#                 push(todo_list, l)
#                 todo_list[link].id = noofurls
#                 push(todo_list[0].outlinks, todo_list[link].id)


def build_nx_graph(graph: dict, url_ids: dict) -> nx.DiGraph:
    """
    Convierte el grafo interno {url: [dst_id, ...]} + url_ids {url: id}
    a un DiGraph de NetworkX donde cada nodo es la URL (string).
    """
    id_to_url = {v: k for k, v in url_ids.items()}
 
    G = nx.DiGraph()
 
    # Agregar todos los nodos conocidos
    for url in url_ids:
        G.add_node(url)
 
    # Agregar aristas
    for src_url, dst_ids in graph.items():
        for dst_id in dst_ids:
            dst_url = id_to_url.get(dst_id)
            if dst_url and dst_url != src_url:          # evitar self-loops
                G.add_edge(src_url, dst_url)
 
    return G

def rank_by_pagerank(pagerank):
    """Lista de URLs ordenada de mayor a menor PageRank."""
    return sorted(pagerank, key=pagerank.get, reverse=True)


def rank_by_authority(authorities):
    """Lista de URLs ordenada de mayor a menor Authority (HITS)."""
    return sorted(authorities, key=authorities.get, reverse=True)


def compute_overlap_curve(ranking_a: list[str], ranking_ref: list[str]) -> list[float]:
    """
    Para cada k de 1..N calcula el % de overlap entre
    el top-k de ranking_a y el top-k de ranking_ref.
    """
    n = min(len(ranking_a), len(ranking_ref))
    overlaps = []
    set_ref = set()
    set_a   = set()

    for k in range(1, n + 1):
        set_a.add(ranking_a[k - 1])
        set_ref.add(ranking_ref[k - 1])
        overlap_pct = len(set_a & set_ref) / k * 100
        overlaps.append(overlap_pct)

    return overlaps


def plot_overlap(overlap_pr: list[float]):
    """
    Grafica la evolución del % de overlap de PageRank y BFS
    respecto del ranking por Authority.
    """
    x = list(range(1, len(overlap_pr) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, overlap_pr, label="PageRank vs Authority", linewidth=2, color="#2196F3")

    ax.set_xlabel("Cantiad de paginas", fontsize=12)
    ax.set_ylabel("Overlap", fontsize=12)
    ax.set_title("Evolucion del overlap de crawl PR vs Auth", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("overlap_evolution.png", dpi=150)
    plt.show()
    print("Gráfico guardado en overlap_evolution.png")

def main():
    initial_urls = [
        "https://www.google.com",
        "https://www.youtube.com",
        "https://mail.google.com",
        "https://docs.google.com",
        "https://www.facebook.com",
        "https://outlook.office.com",
        "https://chatgpt.com",
        "https://login.microsoftonline.com",
        "https://outlook.cloud.microsoft",
        "https://accounts.google.com",
        "https://drive.google.com",
        "https://www.instagram.com",
        "https://x.com",
        "https://github.com",
        "https://gemini.google.com",
        "https://calendar.google.com",
        "https://web.whatsapp.com"
    ]

    graph = {}
    url_ids = {}

    try:
        graph, url_ids = load_graph()
        print("Graph loaded from file.")
    
    except:
        max_page_per_domain = 2000
        max_logical_depth = 5
        max_physical_depth = 10
        max_global_pages = 500

        graph, url_ids = crawl(initial_urls, max_page_per_domain, max_logical_depth, max_physical_depth, max_global_pages)

        save_graph(graph, url_ids)

    # build nx graph
    G = build_nx_graph(graph, url_ids)
    print(f"Nodos: {G.number_of_nodes()}  |  Aristas: {G.number_of_edges()}")

    pagerank = nx.pagerank(G, max_iter=200, tol=1e-6)
    hubs, authorities = nx.hits(G, max_iter=200, tol=1e-6)


    print("\n── Top-10 PageRank ──────────────────────")
    for i, url in enumerate(rank_by_pagerank(pagerank)[:10], 1):
        print(f"  {i:2d}. {url:<60s}  PR={pagerank[url]:.6f}")

    print("\n── Top-10 Authority (HITS) ──────────────")
    for i, url in enumerate(rank_by_authority(authorities)[:10], 1):
        print(f"  {i:2d}. {url:<60s}  Auth={authorities[url]:.6f}")

    # rankings
    ranking_pr   = rank_by_pagerank(pagerank)
    ranking_auth = rank_by_authority(authorities)

    overlap_pr  = compute_overlap_curve(ranking_pr,  ranking_auth)

    plot_overlap(overlap_pr)

    visualize_graph(graph, url_ids)

if __name__ == "__main__":
    main()