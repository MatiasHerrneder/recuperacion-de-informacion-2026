import re
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')

def tokenizer(text: str) -> list[str]:
    """
    Implements the tokenization logic.
    """

    EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    emails = EMAIL_REGEX.findall(text)
    remaining = EMAIL_REGEX.sub("", text)

    # URL_REGEX = re.compile(r'[a-zA-Z]+://[^\s<>"\']+(?<![.,;:!?)\'"])')
    URL_REGEX = re.compile(r'[a-zA-Z]+://[-a-zA-Z0-9@:%._\+~#=]+(?:\.[a-zA-Z0-9()]+)+(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    urls = URL_REGEX.findall(remaining)
    remaining = URL_REGEX.sub("", remaining)

    DATE_REGEX = re.compile(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b')
    dates = DATE_REGEX.findall(remaining)
    remaining = DATE_REGEX.sub("", remaining)

    PHONE_REGEX = re.compile(r'(?:\+\d{1,3}[ -])?\(?\d{2,4}\)?[ -]\d{3,4}(?:[ -]\d{3,4})?')
    phones = PHONE_REGEX.findall(remaining)
    remaining = PHONE_REGEX.sub("", remaining)

    NUMBER_REGEX = re.compile(r'\b\d+(?:[\.,]\d+)?\b')
    numbers = NUMBER_REGEX.findall(remaining)
    remaining = NUMBER_REGEX.sub("", remaining)

    ABREVIATURES_REGEX = re.compile(r'\b(?:[A-Z][a-z]*\.)+')
    abrevitures = ABREVIATURES_REGEX.findall(remaining)
    remaining = ABREVIATURES_REGEX.sub("", remaining)

    ACRONYM_REGEX = re.compile(r'\b[A-Z]{2,}\b')
    acronyms = ACRONYM_REGEX.findall(remaining)
    remaining = ACRONYM_REGEX.sub("", remaining)

    PROPER_NAME_REGEX = re.compile(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b')
    names = PROPER_NAME_REGEX.findall(remaining)
    remaining = PROPER_NAME_REGEX.sub("", remaining)

    remaining = re.findall(r'[^\W\d_]+', remaining.lower())

    stemmer = PorterStemmer()
    remaining = [stemmer.stem(term) for term in remaining]

    return emails + urls + dates + phones + numbers + abrevitures + acronyms + names + remaining