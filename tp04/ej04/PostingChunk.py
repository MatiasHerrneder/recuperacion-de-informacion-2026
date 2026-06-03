from dataclasses import dataclass, field
import struct
from typing import ClassVar, Tuple, BinaryIO


@dataclass
class PostingChunk:
    '''
    Clase para manejo de Chunks (term_id, docid, freq).
    Cada registro ocupa 3 x 4 = 12 bytes en formato big-endian.

    '''

    filename: str

    chunk_id: int = 0
    term_id: int | None = 0
    docid: int = 0
    freq: int = 0
    seek: int = 0

    _file_pointer: BinaryIO = field(init=False, repr=False)
    _NUM_PER_RECORD: ClassVar[int] = 3 # term_id, docid, freq
    _LEN_NUM: ClassVar[int] = 4 # bytes por entero (formato I)

    def __post_init__(self):
        " Inicializar el descriptor de archivo y leer el primer registro"
        self._file_pointer = open(self.filename, 'rb')
        self._read_record()

    def _read_record(self) -> None:
        "Lee un registro basado en seek utilizando unpack"
        self._file_pointer.seek(self.seek)
        raw = self._file_pointer.read(self._NUM_PER_RECORD * self._LEN_NUM)
        
        if len(raw) < self._NUM_PER_RECORD * self._LEN_NUM: # EOF o chunk incompleto
            self.term_id = None
            return
        
        self.term_id, self.docid, self.freq = struct.unpack('>3I', raw)

    def next(self) -> None:
        "Mover el puntero seek al siguente registro"
        self.seek += self._NUM_PER_RECORD * self._LEN_NUM
        self._read_record()

    def get_record(self) -> Tuple[int | None, int, int]:
        "Retorna una tupla con (term_id, doc_id, freq)"
        return (self.term_id, self.docid, self.freq)