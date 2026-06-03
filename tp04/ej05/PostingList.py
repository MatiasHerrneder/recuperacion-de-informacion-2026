from __future__ import annotations
from abc import ABC, abstractmethod
import struct
from typing import List, Optional
from dataclasses import dataclass

class PostingList(ABC):

    @abstractmethod
    def docid(self) -> Optional[int]:
        '''docID actual; None si cursor = -1 (lista agotada).'''
        pass

    @abstractmethod
    def weight(self) -> Optional[float]:
        '''Peso del documento actual (TF-IDF, BM25, etc.).'''
        pass

    @abstractmethod
    def next(self) -> None:
        '''Avanza al siguiente documento. Cursor → -1 si no hay más.'''
        pass

    @abstractmethod
    def ge(self, docid: int) -> Optional[int]:
        '''
        Retorna el primer docID >= docid usando la estrategia configurada.
        Mueve el cursor a esa posición (o a -1 si no existe).
        '''
        pass

    @abstractmethod
    def reset(self) -> None:
        '''Reinicia el cursor al inicio de la lista.'''
        pass

    @abstractmethod
    def posting_and(self, other: PostingList) -> PostingList:
        '''Retorna una nueva PostingList con la intersección de a y b.'''
        pass
        
    @abstractmethod
    def posting_or(self, other: PostingList) -> PostingList:
        '''Retorna una nueva PostingList con la unión de a y b.'''
        pass

    @abstractmethod
    def posting_not(self, universe: PostingList) -> PostingList:
        '''Retorna una nueva PostingList con los documentos de a que no están en b.'''
        pass


@dataclass
class InMemoryPosting(PostingList):

    docids: List[int]
    weights: List[float]
    cursor: int = 0


    def __post_init__(self):
        if not self.docids:
            self.cursor = -1

    def docid(self) -> Optional[int]:
        if self.cursor == -1:
            return None
        return self.docids[self.cursor]

    def weight(self) -> Optional[float]:
        if self.cursor == -1:
            return None
        return self.weights[self.cursor]

    def next(self) -> None:
        if self.cursor != -1:
            self.cursor += 1
            if self.cursor >= len(self.docids):
                self.cursor = -1

    def ge(self, docid: int) -> Optional[int]:
        if self.cursor == -1:
            return None

        # busqueda lineal
        while self.cursor != -1 and self.docids[self.cursor] < docid:
            self.next()

        return self.docid() if self.cursor != -1 else None

    def reset(self) -> None:
        self.cursor = 0 if self.docids else -1

    def posting_and(self, other: PostingList) -> PostingList:
        '''Retorna una nueva PostingList con la intersección de a y b.'''
        
        if self.docid() is None or other.docid() is None:
            return InMemoryPosting([], [])
        docids = []
        self.reset()
        other.reset()
        a, b = self.docid(), other.docid()
        while a is not None and b is not None:
            if a == b:
                docids.append(a)
                self.next()
                other.next()
            elif a < b:
                self.ge(b)
            else:
                other.ge(a)
            a, b = self.docid(), other.docid()
        
        return InMemoryPosting(docids, [1.0] * len(docids))  # pesos dummy
        

    def posting_or(self, other: PostingList) -> PostingList:
        '''Retorna una nueva PostingList con la unión de a y b.'''
        docids = []
        self.reset()
        other.reset()
        a, b = self.docid(), other.docid()
        while a is not None or b is not None:
            if a is not None and (b is None or a < b):
                docids.append(a)
                self.next()
            elif b is not None and (a is None or b < a):
                docids.append(b)
                other.next()
            else:  # a == b
                docids.append(a)
                self.next()
                other.next()
            a, b = self.docid(), other.docid()
        return InMemoryPosting(docids, [1.0] * len(docids))  # pesos dummy


    def posting_not(self, universe: PostingList) -> PostingList:
        result = []
        self.reset()
        universe.reset()

        while (u := universe.docid()) is not None:
            a = self.ge(u)

            if a != u:
                result.append(u)
            universe.next()

        return InMemoryPosting(result, [1.0] * len(result))
    

@dataclass
class DiskPostingList(PostingList):
    '''
    Implementa ge con skiplists
    '''

    index_file: str
    seek: int
    length: int
    skips: dict
    cursor: int = 0

    ENTRY_SIZE = 8

    def _read_entry(self, index: int) -> tuple[int, int]:
        with open(self.index_file, 'rb') as f:
            f.seek(self.seek + index * self.ENTRY_SIZE)
            doc_id, freq = struct.unpack('>II', f.read(self.ENTRY_SIZE))
        return doc_id, freq

    def docid(self) -> Optional[int]:
        if self.cursor == -1:
            return None
        return self._read_entry(self.cursor)[0]

    def weight(self) -> Optional[float]:
        if self.cursor == -1:
            return None
        return float(self._read_entry(self.cursor)[1])

    def next(self) -> None:
        if self.cursor != -1:
            self.cursor += 1
            if self.cursor >= self.length:
                self.cursor = -1

    def ge(self, docid: int) -> Optional[int]:
        if self.cursor == -1:
            return None

        while self.cursor in self.skips:
            dst_index = self.skips[self.cursor]
            dst_docid = self._read_entry(dst_index)[0]
            if dst_docid <= docid:
                self.cursor = dst_index
            else:
                break

        while self.cursor != -1 and self._read_entry(self.cursor)[0] < docid:
            self.next()

        return self.docid()

    def reset(self) -> None:
        self.cursor = 0

    def posting_and(self, other: PostingList) -> PostingList:
        '''Retorna una nueva PostingList con la intersección de a y b.'''
        
        if self.docid() is None or other.docid() is None:
            return InMemoryPosting([], [])
        docids = []
        self.reset()
        other.reset()
        a, b = self.docid(), other.docid()
        while a is not None and b is not None:
            if a == b:
                docids.append(a)
                self.next()
                other.next()
            elif a < b:
                self.ge(b)
            else:
                other.ge(a)
            a, b = self.docid(), other.docid()
        
        return InMemoryPosting(docids, [1.0] * len(docids))  # pesos dummy
        

    def posting_or(self, other: PostingList) -> PostingList:
        '''Retorna una nueva PostingList con la unión de a y b.'''
        docids = []
        self.reset()
        other.reset()
        a, b = self.docid(), other.docid()
        while a is not None or b is not None:
            if a is not None and (b is None or a < b):
                docids.append(a)
                self.next()
            elif b is not None and (a is None or b < a):
                docids.append(b)
                other.next()
            else:  # a == b
                docids.append(a)
                self.next()
                other.next()
            a, b = self.docid(), other.docid()
        return InMemoryPosting(docids, [1.0] * len(docids))  # pesos dummy


    def posting_not(self, universe: PostingList) -> PostingList:
        result = []
        self.reset()
        universe.reset()

        while (u := universe.docid()) is not None:
            a = self.ge(u)

            if a != u:
                result.append(u)
            universe.next()

        return InMemoryPosting(result, [1.0] * len(result))