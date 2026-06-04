import argparse
import os
import pickle
import struct
import time
import math
from typing import List, Tuple

INDEX_SUBPATH          = "index.bin"
VOCABULARY_SUBPATH     = "vocabulary.pkl"
TERM_ID_SUBPATH        = "term_index.pkl"
DOCUMENT_INDEX_SUBPATH = "document_index.pkl"



# VByte

def vbyte_encode(n: int) -> bytes:
    result = []
    while True:
        result.insert(0, n & 0x7F)
        n >>= 7
        if n == 0:
            break
    result[-1] |= 0x80
    return bytes(result)


def vbyte_decode(data: bytes, offset: int) -> Tuple[int, int]:
    """
    Decodifica un entero VByte desde data[offset]. Devuelve (valor, nuevo_offset).
    Convencion: el ultimo byte tiene el bit 0x80 seteado (igual que vbyte_encode).
    Los bytes anteriores tienen 0x80 en 0 (bit de continuacion).
    Los datos van de mas significativo a menos significativo.
    """
    # recolectar bytes hasta encontrar el que tiene 0x80
    raw = []
    while True:
        byte = data[offset]; offset += 1
        raw.append(byte)
        if byte & 0x80:   # este es el ultimo byte
            break
    # decodificar: raw[0] es el mas significativo
    value = 0
    for b in raw:
        value = (value << 7) | (b & 0x7F)
    return value, offset



# elias-gamma

class BitWriter:
    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bits_in_current = 0

    def write_bits(self, value: int, n_bits: int):
        for i in range(n_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.current_byte = (self.current_byte << 1) | bit
            self.bits_in_current += 1
            if self.bits_in_current == 8:
                self.buffer.append(self.current_byte)
                self.current_byte = 0
                self.bits_in_current = 0

    def flush(self) -> bytes:
        if self.bits_in_current > 0:
            self.current_byte <<= (8 - self.bits_in_current)
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bits_in_current = 0
        return bytes(self.buffer)


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7

    def read_bit(self) -> int:
        if self.byte_pos >= len(self.data):
            raise EOFError("Sin mas bits")
        bit = (self.data[self.byte_pos] >> self.bit_pos) & 1
        self.bit_pos -= 1
        if self.bit_pos < 0:
            self.bit_pos = 7
            self.byte_pos += 1
        return bit


def elias_gamma_encode_all(values: List[int]) -> bytes:
    """Codifica una lista de enteros >= 1 con Elias-gamma."""
    writer = BitWriter()
    for n in values:
        if n < 1:
            raise ValueError(f"Elias-gamma requiere n >= 1, recibido: {n}")
        k = n.bit_length() - 1
        writer.write_bits(0, k)
        writer.write_bits(n, k + 1)
    return writer.flush()


def elias_gamma_decode_all(data: bytes, count: int) -> List[int]:
    """Decodifica `count` enteros Elias-gamma desde data."""
    reader = BitReader(data)
    result = []
    for _ in range(count):
        k = 0
        while reader.read_bit() == 0:
            k += 1
        value = 1
        for _ in range(k):
            value = (value << 1) | reader.read_bit()
        result.append(value)
    return result



# Lectura del indice original

def read_original_posting(index_bytes: bytes, seek: int, length: int) -> List[Tuple[int, int]]:
    """Lee una posting list del índice original (sin comprimir)."""
    data = index_bytes[seek: seek + length * 8]
    return [struct.unpack('>II', data[i:i+8]) for i in range(0, len(data), 8)]


def compress_posting(posting: List[Tuple[int, int]], use_dgaps: bool) -> Tuple[bytes, bytes]:
    """
    Devuelve (docids_bytes, freqs_bytes) en streams totalmente independientes.
    docids → VByte puro (sin BitWriter), con o sin DGaps
    freqs  → Elias-gamma en su propio BitWriter
    """
    doc_ids = [d for d, _ in posting]
    freqs   = [f for _, f in posting]

    
    if use_dgaps:
        prev = 0
        encoded_ids = []
        for d in doc_ids:
            encoded_ids.append(vbyte_encode(d - prev))
            prev = d
    else:
        encoded_ids = [vbyte_encode(d) for d in doc_ids]
    docids_bytes = b"".join(encoded_ids)


    freqs_bytes = elias_gamma_encode_all(freqs)

    return docids_bytes, freqs_bytes


def decompress_posting(
    docids_bytes: bytes, freqs_bytes: bytes,
    length: int, use_dgaps: bool
) -> List[Tuple[int, int]]:
    """Decodifica una posting list comprimida."""
    decoded_ids = []
    offset = 0
    for _ in range(length):
        val, offset = vbyte_decode(docids_bytes, offset)
        decoded_ids.append(val)

    if use_dgaps:
        prev = 0
        abs_ids = []
        for gap in decoded_ids:
            prev += gap
            abs_ids.append(prev)
        decoded_ids = abs_ids

    decoded_freqs = elias_gamma_decode_all(freqs_bytes, length)

    return list(zip(decoded_ids, decoded_freqs))


def build_compressed_index(
    index_bytes: bytes,
    vocabulary: dict,
    output_dir: str,
    use_dgaps: bool,
) -> dict:
    """
    Escribe docids.bin y freqs.bin en output_dir.
    Devuelve el vocabulario comprimido:
      term_id -> [seek_docids, seek_freqs, length, size_docids, size_freqs]
    """
    os.makedirs(output_dir, exist_ok=True)
    compressed_vocab = {}

    with open(os.path.join(output_dir, "docids.bin"), "wb") as f_ids, \
         open(os.path.join(output_dir, "freqs.bin"),  "wb") as f_frq:

        for term_id in sorted(vocabulary):
            seek, length = vocabulary[term_id][:2]
            posting = read_original_posting(index_bytes, seek, length)

            docids_bytes, freqs_bytes = compress_posting(posting, use_dgaps)

            seek_ids = f_ids.tell()
            seek_frq = f_frq.tell()
            f_ids.write(docids_bytes)
            f_frq.write(freqs_bytes)

            compressed_vocab[term_id] = [
                seek_ids, seek_frq, length,
                len(docids_bytes), len(freqs_bytes),
            ]

    pickle.dump(compressed_vocab,
                open(os.path.join(output_dir, "compressed_vocab.pkl"), "wb"))
    return compressed_vocab


def verify_and_time_decompression(
    output_dir: str,
    vocabulary_orig: dict,
    index_bytes: bytes,
    use_dgaps: bool,
) -> Tuple[float, bool]:
    """
    Descomprime todas las posting lists y verifica que sean iguales al original.
    Devuelve (tiempo_us, todo_correcto).
    """
    compressed_vocab = pickle.load(
        open(os.path.join(output_dir, "compressed_vocab.pkl"), "rb"))
    docids_data = open(os.path.join(output_dir, "docids.bin"), "rb").read()
    freqs_data  = open(os.path.join(output_dir, "freqs.bin"),  "rb").read()

    t0 = time.perf_counter()
    ok = True
    first_error = None
    for term_id in sorted(compressed_vocab):
        seek_ids, seek_frq, length, size_ids, size_frq = compressed_vocab[term_id]

        docids_bytes = docids_data[seek_ids: seek_ids + size_ids]
        freqs_bytes  = freqs_data [seek_frq: seek_frq  + size_frq]

        decoded = decompress_posting(docids_bytes, freqs_bytes, length, use_dgaps)

        seek_orig, length_orig = vocabulary_orig[term_id][:2]
        original = read_original_posting(index_bytes, seek_orig, length_orig)
        if decoded != original:
            if ok:
                # encontrar el primer elemento que difiere
                diff_idx = next((i for i,(a,b) in enumerate(zip(original,decoded)) if a!=b), len(decoded))
                first_error = {
                    "term_id":  term_id,
                    "length":   length,
                    "diff_idx": diff_idx,
                    "original": original[max(0,diff_idx-1):diff_idx+3],
                    "decoded":  decoded [max(0,diff_idx-1):diff_idx+3],
                }
            ok = False
    t1 = time.perf_counter()

    if first_error:
        e = first_error
        print(f"\n  [DEBUG] Primer mismatch en term_id={e['term_id']} (length={e['length']}), en indice {e['diff_idx']}")
        print(f"    original alrededor = {e['original']}")
        print(f"    decoded  alrededor = {e['decoded']}")

    return (t1 - t0) * 1_000, ok   # ms


def run(index_path: str, output_base: str):
    print("=" * 65)
    print("  COMPRESION DE INDICE: VByte (docIDs) + Elias-gamma (freqs)")
    print("=" * 65)

    # cargar indice original
    vocabulary = pickle.load(open(os.path.join(index_path, VOCABULARY_SUBPATH), "rb"))
    term_to_id = pickle.load(open(os.path.join(index_path, TERM_ID_SUBPATH),    "rb"))
    index_bytes = open(os.path.join(index_path, INDEX_SUBPATH), "rb").read()

    original_size = len(index_bytes)
    n_terms = len(vocabulary)
    n_docs_total = sum(v[1] for v in vocabulary.values())

    print(f"\n  Terminos:          {n_terms:>10,}")
    print(f"  Postings totales:  {n_docs_total:>10,}")
    print(f"  Tamano original:   {original_size:>10,} bytes  ({original_size/1024:.1f} KB)")

    experiments = [
        ("CON DGaps",  True,  os.path.join(output_base, "with_dgaps")),
        ("SIN DGaps",  False, os.path.join(output_base, "no_dgaps")),
    ]

    rows = []
    for label, use_dgaps, out_dir in experiments:
        print(f"\n  {'─'*50}")
        print(f"  Experimento: {label}")
        print(f"  {'─'*50}")

        # compresion
        t0 = time.perf_counter()
        compressed_vocab = build_compressed_index(index_bytes, vocabulary, out_dir, use_dgaps)
        t_compress = (time.perf_counter() - t0) * 1_000  # ms

        size_ids = os.path.getsize(os.path.join(out_dir, "docids.bin"))
        size_frq = os.path.getsize(os.path.join(out_dir, "freqs.bin"))
        size_total = size_ids + size_frq

        ratio = size_total / original_size * 100

        print(f"  Tiempo compresion:    {t_compress:>9.2f} ms")
        print(f"  docids.bin:           {size_ids:>9,} bytes  ({size_ids/1024:.1f} KB)")
        print(f"  freqs.bin:            {size_frq:>9,} bytes  ({size_frq/1024:.1f} KB)")
        print(f"  Total comprimido:     {size_total:>9,} bytes  ({size_total/1024:.1f} KB)")
        print(f"  Ratio vs original:    {ratio:>9.1f}%  (factor {original_size/size_total:.2f}x)")

        # descompresion + verificacion
        t_decomp, ok = verify_and_time_decompression(
            out_dir, vocabulary, index_bytes, use_dgaps)
        status = "OK" if ok else "ERROR - resultados no coinciden"
        print(f"  Tiempo descompresion: {t_decomp:>9.2f} ms  [{status}]")

        rows.append((label, t_compress, t_decomp, size_ids, size_frq, size_total, ratio))

    # tabla comparativa
    print(f"\n{'='*65}")
    print("  COMPARATIVA")
    print(f"{'='*65}")
    print(f"  {'':18} {'CON DGaps':>16} {'SIN DGaps':>16}")
    print(f"  {'-'*52}")
    fields = [
        ("T. compresion (ms)",   1, "{:.2f}"),
        ("T. descompresion (ms)",2, "{:.2f}"),
        ("docids.bin (bytes)",   3, "{:,}"),
        ("freqs.bin (bytes)",    4, "{:,}"),
        ("Total (bytes)",        5, "{:,}"),
        ("Ratio vs original",    6, "{:.1f}%"),
    ]
    for name, idx, fmt in fields:
        v1 = fmt.format(rows[0][idx])
        v2 = fmt.format(rows[1][idx])
        print(f"  {name:<22} {v1:>16} {v2:>16}")

    print(f"\n  Tamano original: {original_size:,} bytes ({original_size/1024:.1f} KB)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprime el indice con VByte + Elias-gamma")
    parser.add_argument("index_path",  help="Directorio del indice original (output2/)")
    parser.add_argument("--output_dir", default="compressed",
                        help="Directorio base para los indices comprimidos (default: compressed/)")
    args = parser.parse_args()

    run(args.index_path, args.output_dir)