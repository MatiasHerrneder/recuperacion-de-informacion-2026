import argparse
import os
import pickle
import struct
import re
from tokenizer import tokenizer
from PostingChunk import PostingChunk
import shutil


def index_bsbi(corpus_path: str, memory_limit: int = 10, output_dir: str = "output", stop_words_path: str | None = None):
    '''
    Implementa el algoritmo BSBI para indexar un corpus de documentos
    corpus_path: ruta al directorio con el corpus
    memory_limit: numero maximo de documentos a procesar en memoria antes de escribir un bloque a disco
    output_dir: ruta al directorio de salida
    stop_words_path: ruta al archivo con las stopwords
    '''
    
    MIN_TERM_LENGTH: int = 3
    MAX_TERM_LENGTH: int = 100

    INDEX_SUBPATH: str = "index.bin"
    VOCABULARY_SUBPATH: str = "vocabulary.pkl"
    TERM_ID_SUBPATH: str = "term_index.pkl"
    DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir)

    term_to_id = {}
    memory_counter = 0
    partial_tuples = []
    chunk_id = 0

    doc_index = {}
    seen_files = set()
    stop_words = set()
    
    if stop_words_path and os.path.isfile(stop_words_path):
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            stop_words = set(re.findall(r'[a-z]+', f.read().lower()))

    TXT_FILE_REGEX = re.compile(r"\.txt$", re.IGNORECASE)

    if os.path.isdir(corpus_path):
        files_to_process = os.walk(corpus_path)
    else:
        if os.path.isfile(corpus_path) and TXT_FILE_REGEX.search(corpus_path):
            files_to_process = [(os.path.dirname(corpus_path), [], [os.path.basename(corpus_path)])]
        else:
            raise Exception(f"Error: La ruta '{corpus_path}' no es un directorio o archivo válido.")

    for root, _, files in files_to_process:
        for file in files:
            
            filepath = os.path.join(root, file)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                    doc_key = os.path.basename(filepath)
                    if doc_key in seen_files:
                        raise Exception(f"Archivo '{filepath}' ya procesado, no se permiten nombres de archivos repetidos.")
                    seen_files.add(doc_key)
                    doc_id = len(doc_index)
                    doc_index[doc_id] = doc_key

                    words = tokenizer(text)
                    doc_terms = {}
                    
                    for word in words:
                        if word not in stop_words and MIN_TERM_LENGTH <= len(word) <= MAX_TERM_LENGTH:
                            if word not in doc_terms:
                                doc_terms[word] = {}
                            doc_terms[word][doc_id] = doc_terms[word].get(doc_id, 0) + 1
                            if word not in term_to_id:
                                term_to_id[word] = len(term_to_id)
                    
                    for term, doc_freq in doc_terms.items():
                        for doc_id, freq in doc_freq.items():
                            partial_tuples.append((term_to_id[term], doc_id, freq))
                    
                    memory_counter += 1

                    if memory_counter >= memory_limit:
                        partial_tuples.sort(key=lambda x: (x[0], x[1]))
                        # partial_tuples.sort(key=lambda x: (x[0]))
                        flat = [item for tup in partial_tuples for item in tup]
                        with open(os.path.join(output_dir, "chunks", f"chunk_{chunk_id}.bin"), 'wb') as bin:
                            bin.write(struct.pack(f'>{len(flat)}I', *flat))
                        chunk_id += 1
                        partial_tuples = []
                        memory_counter = 0

            except Exception as e:
                print(f"Error al leer el archivo '{filepath}': {e}")

    if partial_tuples:
        # partial_tuples.sort(key=lambda x: (x[0]))
        partial_tuples.sort(key=lambda x: (x[0], x[1]))
        flat = [item for tup in partial_tuples for item in tup]
        with open(os.path.join(output_dir, "chunks", f"chunk_{chunk_id}.bin"), 'wb') as bin:
            bin.write(struct.pack(f'>{len(flat)}I', *flat))
        chunk_id += 1
    

    # MERGE
    chunk_pointers = [PostingChunk(os.path.join(output_dir, "chunks", f"chunk_{i}.bin")) for i in range(chunk_id)]

    vocabulary = {}

    with open(os.path.join(output_dir, INDEX_SUBPATH), 'wb') as index:
        for term_id_actual in sorted(term_to_id.values()):
            posting_lists = []
            for chunk in chunk_pointers:
                while chunk.term_id is not None and chunk.term_id < term_id_actual:
                    chunk.next()
                while chunk.term_id is not None and chunk.term_id == term_id_actual:
                    _, doc_id, freq = chunk.get_record()
                    posting_lists.append((doc_id, freq))
                    chunk.next()

            if posting_lists:
                posting_lists.sort(key=lambda x: x[0])  # asegurar orden por doc_id
                seek_actual = index.tell()
                encoded = encode_posting_list(posting_lists)
                index.write(encoded)
                # guardamos: [offset_en_bytes, cantidad_de_docs, tamaño_en_bytes]
                vocabulary[term_id_actual] = [seek_actual, len(posting_lists), len(encoded)]

    pickle.dump(vocabulary, open(os.path.join(output_dir, VOCABULARY_SUBPATH), 'wb'))
    pickle.dump(term_to_id, open(os.path.join(output_dir, TERM_ID_SUBPATH), 'wb'))
    pickle.dump(doc_index, open(os.path.join(output_dir, DOCUMENT_INDEX_SUBPATH), 'wb'))

    # implementar que pueda retomar si se cae ??

def vbyte_encode(n: int) -> bytes:
    """Codifica un entero positivo con Variable Byte encoding."""
    result = []
    while True:
        result.insert(0, n & 0x7F)
        n >>= 7
        if n == 0:
            break
    result[-1] |= 0x80
    return bytes(result)

def vbyte_encode_posting_list(posting_list: list[tuple[int, int]]) -> bytes:
    """
    Recibe lista de (doc_id, freq) ya ordenada por doc_id.
    Devuelve los bytes con d-gaps en doc_ids y freqs sin comprimir.
    """
    out = bytearray()
    prev_doc_id = 0
    for doc_id, freq in posting_list:
        gap = doc_id - prev_doc_id   # d-gap
        out += vbyte_encode(gap)
        out += vbyte_encode(freq)    # freq también VByte (son números chicos)
        prev_doc_id = doc_id
    return bytes(out)

def vbyte_decode_posting_list(data: bytes, n_docs: int) -> list[tuple[int, int]]:
    """Decodifica una posting list VByte. n_docs = cantidad de pares esperados."""
    result = []
    i = 0
    prev_doc_id = 0
    for _ in range(n_docs):
        # decodificar gap
        gap = 0
        shift = 0
        while True:
            byte = data[i]; i += 1
            gap |= (byte & 0x7F) << shift
            shift += 7
            if byte & 0x80:  # bit de continuación = es el último byte
                break
        # decodificar freq
        freq = 0
        shift = 0
        while True:
            byte = data[i]; i += 1
            freq |= (byte & 0x7F) << shift
            shift += 7
            if byte & 0x80:
                break
        prev_doc_id += gap
        result.append((prev_doc_id, freq))
    return result

class BitWriter:
    """Acumula bits y los escribe a bytes cuando están listos."""
    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bits_in_current = 0

    def write_bits(self, value: int, n_bits: int):
        """Escribe los n_bits menos significativos de value."""
        for i in range(n_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.current_byte = (self.current_byte << 1) | bit
            self.bits_in_current += 1
            if self.bits_in_current == 8:
                self.buffer.append(self.current_byte)
                self.current_byte = 0
                self.bits_in_current = 0

    def flush(self) -> bytes:
        """Rellena con ceros y devuelve todos los bytes."""
        if self.bits_in_current > 0:
            self.current_byte <<= (8 - self.bits_in_current)
            self.buffer.append(self.current_byte)
        return bytes(self.buffer)

class BitReader:
    """Lee bits uno a uno desde un bytearray."""
    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7  # bit más significativo primero

    def read_bit(self) -> int:
        if self.byte_pos >= len(self.data):
            raise EOFError("No hay más bits")
        bit = (self.data[self.byte_pos] >> self.bit_pos) & 1
        self.bit_pos -= 1
        if self.bit_pos < 0:
            self.bit_pos = 7
            self.byte_pos += 1
        return bit

def elias_gamma_encode(n: int, writer: BitWriter):
    """
    Elias-gamma para n >= 1.
    Escribe floor(log2(n)) ceros, luego n en binario.
    """
    if n < 1:
        raise ValueError("Elias-gamma requiere n >= 1")
    k = n.bit_length() - 1   # floor(log2(n))
    writer.write_bits(0, k)   # k ceros (unary prefix)
    writer.write_bits(n, k + 1)  # n en binario con k+1 bits

def elias_gamma_decode(reader: BitReader) -> int:
    """Lee un número codificado en Elias-gamma."""
    k = 0
    while reader.read_bit() == 0:
        k += 1
    # leer los k bits restantes (el primero ya fue leído y es 1)
    value = 1
    for _ in range(k):
        value = (value << 1) | reader.read_bit()
    return value

def encode_posting_list(posting_list: list[tuple[int, int]]) -> bytes:
    """
    doc_ids  → VByte con d-gaps
    freqs    → Elias-gamma
    Todo mezclado en un bytearray híbrido.
    """
    writer = BitWriter()
    prev_doc_id = 0
    for doc_id, freq in posting_list:
        gap = doc_id - prev_doc_id
        # VByte para el gap: escribimos byte a byte en el BitWriter
        for byte in vbyte_encode(gap):
            writer.write_bits(byte, 8)
        # Elias-gamma para la frecuencia
        elias_gamma_encode(freq, writer)
        prev_doc_id = doc_id
    return writer.flush()

def decode_posting_list(data: bytes, n_docs: int) -> list[tuple[int, int]]:
    reader = BitReader(data)
    result = []
    prev_doc_id = 0
    for _ in range(n_docs):
        # leer bytes VByte para el gap
        # vbyte_encode pone los bytes de más significativo a menos significativo
        # el último byte tiene bit 0x80 seteado
        raw_bytes = []
        while True:
            byte = 0
            for _ in range(8):
                byte = (byte << 1) | reader.read_bit()
            raw_bytes.append(byte)
            if byte & 0x80:  # último byte
                break

        # decodificar: los bytes vienen de más a menos significativo
        gap = 0
        for b in raw_bytes:
            gap = (gap << 7) | (b & 0x7F)

        freq = elias_gamma_decode(reader)
        prev_doc_id += gap
        result.append((prev_doc_id, freq))
    return result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_path", help="Directory with the files to index (scans recursively)")
    parser.add_argument("--memory_limit", type=int, help="Memory limit for buffering documents before writing to disk")
    parser.add_argument("--output_path", help="Output directory for index files")
    parser.add_argument("--stop_words_path", help="Path to file with stop words")
    args = parser.parse_args()

    index_bsbi(args.corpus_path)


if __name__ == "__main__":
    main()
    # test = [(0, 3), (5, 1), (100, 7), (101, 2), (5000, 10)]
    # encoded = encode_posting_list(test)
    # decoded = decode_posting_list(encoded, len(test))
    # print("original:", test)
    # print("decoded: ", decoded)
    # assert test == decoded, "MISMATCH"
    # print("OK")