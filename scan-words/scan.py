import os
from collections import Counter
import re

def scan_words(dir: str) -> dict[str, int]:
    """
    Scan a directory for txt files and return a list of all words found and their frequencies.
    Arguments:
        dir (string): The directory to scan for txt files.
    Returns:
        A dictionary with words as keys and their frequencies as values.
    """

    word_freq = Counter()

    for root, _, files in os.walk(dir):
        for file in files:
            if re.search(r"\.txt$", file):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    words = re.findall(r'\b\w+\b', f.read().lower())
                    word_freq.update(words)

    return dict(word_freq)