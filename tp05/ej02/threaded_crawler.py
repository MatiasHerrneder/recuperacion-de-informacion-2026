from extract_links import request_and_extract_links
from urllib.parse import urlparse
import pickle
from pyvis.network import Network
import queue
import threading


def crawl(initial_urls: list, max_page_per_domain: int, max_logical_depth: int, max_physical_depth: int, num_threads: int = 8):
    todo_queue = queue.Queue()   # thread-safe, reemplaza tu Queue custom
    visited_urls = set()
    graph = {}
    url_ids = {}
    id_count = [0]               # lista para poder mutarla desde threads
    logical_depth = {}
    pages_per_domain = {}

    lock = threading.Lock()      # protege todas las estructuras compartidas

    seed_domains = {get_domain(url) for url in initial_urls}

    for url in initial_urls:
        todo_queue.put(url)
        url_ids[url] = id_count[0]
        logical_depth[url] = 0
        pages_per_domain[get_domain(url)] = pages_per_domain.get(get_domain(url), 0) + 1
        id_count[0] += 1

    def worker():
        while True:
            try:
                curr_url = todo_queue.get(timeout=5)  # espera 5s antes de rendirse
            except queue.Empty:
                break  # cola vacía por 5s → este thread termina

            print(f"[{threading.current_thread().name}] Crawling: {curr_url} | Queue size: {todo_queue.qsize()}")

            # marcar como visitada antes del fetch
            with lock:
                if curr_url in visited_urls:
                    todo_queue.task_done()
                    continue
                visited_urls.add(curr_url)
                graph[curr_url] = []
                current_logical_depth = logical_depth.get(curr_url, 0)
                current_domain = get_domain(curr_url)

            # fetch fuera del lock: es la operación lenta (I/O)
            try:
                links = request_and_extract_links(curr_url)
            except Exception as e:
                print(f"Error fetching {curr_url}: {e}")
                todo_queue.task_done()
                continue

            # procesar links con lock
            with lock:
                for link in links:
                    if link in visited_urls or link in url_ids:
                        if link in url_ids:
                            graph[curr_url].append(url_ids[link])
                    else:
                        link_domain = get_domain(link)
                        link_logical_depth = current_logical_depth + 1

                        if (passes_url_filter(link) and
                            pages_per_domain.get(link_domain, 0) < max_page_per_domain and
                            link_logical_depth <= max_logical_depth and
                            physical_depth(link) <= max_physical_depth):
                            # physical_depth(link) <= max_physical_depth and
                            # urlparse(link).netloc in seed_domains):

                            logical_depth[link] = link_logical_depth
                            url_ids[link] = id_count[0]
                            id_count[0] += 1
                            pages_per_domain[link_domain] = pages_per_domain.get(link_domain, 0) + 1
                            graph[curr_url].append(url_ids[link])
                            todo_queue.put(link)

            todo_queue.task_done()

    # lanzar threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, name=f"Worker-{i+1}", daemon=True)
        t.start()
        threads.append(t)

    # esperar a que terminen todos
    for t in threads:
        t.join()

    return graph, url_ids


def get_domain(url):
    return urlparse(url).netloc

def physical_depth(url):
    path = urlparse(url).path
    segments = [s for s in path.split("/") if s]
    return len(segments)

def passes_url_filter(url: str) -> bool:
    return urlparse(url).scheme in ("http", "https")


def save_graph(graph, url_ids, filename="graph.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"graph": graph, "url_ids": url_ids}, f)

def load_graph(filename="graph.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["graph"], data["url_ids"]

def visualize_graph(graph, url_ids, filename="graph.html"):
    net = Network(height="750px", width="100%", directed=True)
    net.barnes_hut()

    for url, node_id in url_ids.items():
        label = urlparse(url).netloc
        net.add_node(node_id, label=label, title=url)

    for src_url, outlinks in graph.items():
        src_id = url_ids[src_url]
        for dst_id in outlinks:
            net.add_edge(src_id, dst_id)

    net.show(filename, notebook=False)


def graph_stats(graph, url_ids):
    num_nodes = len(url_ids)
    num_edges = sum(len(outlinks) for outlinks in graph.values())
    
    print(f"Nodos: {num_nodes}")
    print(f"Aristas: {num_edges}")


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
        "https://campus-1001.ammon.cloud",
        "https://www.linkedin.com",
        "https://www.bing.com",
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
        max_page_per_domain = 20
        max_logical_depth = 3
        max_physical_depth = 3
        num_threads = 32

        graph, url_ids = crawl(
            initial_urls,
            max_page_per_domain,
            max_logical_depth,
            max_physical_depth,
            num_threads=num_threads
        )

        save_graph(graph, url_ids)

    graph_stats(graph, url_ids)

    # print("Graph:", graph)
    # print("URL IDs:", url_ids)

    visualize_graph(graph, url_ids)

if __name__ == "__main__":
    main()