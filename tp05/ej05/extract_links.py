import requests
import argparse
from bs4 import BeautifulSoup

def extract_links(text: str) -> list:
    """
    Extracts all links from a given HTML
    Args:
        text (str): HTML content
    Returns:
        list: list of links
    """

    parsed_html = BeautifulSoup(text, 'html.parser')
    links = [link.get('href') for link in parsed_html.find_all('a', href=True)]

    return links


def request_and_extract_links(url: str) -> list:
    """
    Requests a web page and returns its links
    Args:
        url (str): url to request
    Returns:
        list: list of links
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-419,es;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    response = requests.get(url, headers=headers, timeout=(5,10))

    if response.status_code == 200:
        html_content = response.text
        return extract_links(html_content)
    else:
        print(f"Failed to retrieve the URL: {url}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Url to extract links from")
    args = parser.parse_args()

    link_list = request_and_extract_links(args.url)

    for link in link_list:
        print(link)

if __name__ == "__main__":
    main()