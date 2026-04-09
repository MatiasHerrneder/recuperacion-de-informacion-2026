import os
import re
import argparse
import nltk
from nltk.stem import SnowballStemmer

nltk.download('punkt')
ss = SnowballStemmer('spanish')

def lex(dir: str) -> dict[str, list | dict[str, int]]:
    """
    Scan a directory recursively for txt files and return a list of all words terms found and their document frequency, plus some other statistics.
    Arguments:
        dir (string): The directory to scan for txt files.
    Returns:
        A dictionary with terms and their document frequencies, and statistics about the corpus.
    """

    doc_count = 0
    token_count = 0
    dfs = {}

    if not os.path.isdir(dir):
        print(f"Error: La ruta '{dir}' no es un directorio válido.")
        return {}

    for root, _, files in os.walk(dir):
        for file in files:
            if re.search(r"\.txt$", file):
                doc_count += 1
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        words = re.findall(r'[a-z]+', f.read().lower())
                        token_count += len(words)
                        for word in set(words):
                            word = ss.stem(word)
                            dfs[word] = dfs.get(word, 0) + 1
                except Exception as e:
                    print(f"Error al leer el archivo '{file}': {e}")

    return {
        "data": [{"term": term, "df": df} for term, df in dfs.items()],
        "statistics": {
            "num_docs": doc_count,
            "num_terms": len(dfs),
            "num_tokens": token_count
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with the .txt files to process (scans recursively)")
    args = parser.parse_args()

    print(lex(args.dir))

if __name__ == "__main__":
    main()