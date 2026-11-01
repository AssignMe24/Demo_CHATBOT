from __future__ import annotations
from pathlib import Path
import faiss,numpy as np
class VectorStore:
    def __init__(self,embeddings):
        embeddings=np.ascontiguousarray(self._f32(embeddings))
        if embeddings.ndim!=2 or embeddings.shape[0]==0 or embeddings.shape[1]==0: raise ValueError("Embeddings must be a non-empty 2D array.")
        self.dim=int(embeddings.shape[1]);self.index=faiss.IndexFlatIP(self.dim);self.index.add(embeddings)
    def save(self,path): Path(path).parent.mkdir(parents=True,exist_ok=True);faiss.write_index(self.index,str(path))
    @classmethod
    def load(cls,path):
        inst=cls.__new__(cls);inst.index=faiss.read_index(str(path));inst.dim=int(inst.index.d);return inst
    def search(self,query_embedding,top_k):
        q=np.ascontiguousarray(self._f32(query_embedding));q=q.reshape(1,-1) if q.ndim==1 else q
        if q.ndim!=2 or q.shape[1]!=self.dim: raise ValueError(f"Query embedding dimension does not match index dimension {self.dim}.")
        top_k=max(0,min(int(top_k),int(self.index.ntotal)))
        if top_k==0:return np.asarray([],dtype=np.int64),np.asarray([],dtype=np.float32)
        scores,indices=self.index.search(q,top_k);valid=indices[0]>=0;return indices[0][valid],scores[0][valid]
    def reconstruct(self,index):
        if index<0 or index>=self.index.ntotal: raise IndexError(index)
        return np.asarray(self.index.reconstruct(int(index)),dtype=np.float32)
    @staticmethod
    def _f32(a): return a.astype(np.float32) if a.dtype!=np.float32 else a
