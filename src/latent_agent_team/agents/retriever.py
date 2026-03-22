"""
retriever.py — Retriever agent: memory search + FAISS-backed dense retrieval.

Responsibilities:
  1. Maintain a FAISS index of past observations, DOM snippets, product descriptions
  2. Encode queries using the shared backbone
  3. Retrieve top-K relevant passages for the current task context
  4. Report retrieval confidence (used by AdaptiveBitrateScheduler)
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..models.backbone import BackboneManager
from ..models.role_adapter import RoleAdapterManager
from ..models.latent_comm import LatentCommunicationModule

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not found — retriever will use brute-force cosine search")


@dataclass
class RetrievalResult:
    index: int
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FAISSMemoryIndex:
    """
    FAISS-backed dense retrieval index.
    Stores text passages + their embeddings for fast ANN search.
    """

    def __init__(self, embedding_dim: int = 3072, nlist: int = 100):
        self.embedding_dim = embedding_dim
        self.texts: List[str] = []
        self.metadata: List[Dict] = []

        if FAISS_AVAILABLE:
            # IVF index for large corpora, flat for small
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner-product (cosine after normalize)
        else:
            self.embeddings: List[np.ndarray] = []

    def add(
        self,
        texts: List[str],
        embeddings: np.ndarray,  # [N, embedding_dim]
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Add passages to the index."""
        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        normalized = embeddings / norms

        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

        if FAISS_AVAILABLE:
            self.index.add(normalized.astype(np.float32))
        else:
            self.embeddings.extend(list(normalized))

    def search(
        self,
        query_embedding: np.ndarray,  # [embedding_dim]
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve top-K passages for a query embedding."""
        norm = np.linalg.norm(query_embedding) + 1e-10
        query_norm = (query_embedding / norm).astype(np.float32)

        if FAISS_AVAILABLE and len(self.texts) > 0:
            scores, indices = self.index.search(
                query_norm.reshape(1, -1), min(top_k, len(self.texts))
            )
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.texts):
                    results.append(
                        RetrievalResult(
                            index=int(idx),
                            score=float(score),
                            text=self.texts[idx],
                            metadata=self.metadata[idx],
                        )
                    )
            return results
        elif not FAISS_AVAILABLE and len(self.texts) > 0:
            # Brute-force cosine similarity
            emb_matrix = np.stack(self.embeddings, axis=0)
            scores = emb_matrix @ query_norm
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [
                RetrievalResult(
                    index=int(i),
                    score=float(scores[i]),
                    text=self.texts[i],
                    metadata=self.metadata[i],
                )
                for i in top_indices
            ]
        return []

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if FAISS_AVAILABLE:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        else:
            with open(os.path.join(path, "embeddings.pkl"), "wb") as f:
                pickle.dump(self.embeddings, f)
        with open(os.path.join(path, "texts.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)

    def load(self, path: str) -> None:
        if FAISS_AVAILABLE and os.path.exists(os.path.join(path, "index.faiss")):
            self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        elif os.path.exists(os.path.join(path, "embeddings.pkl")):
            with open(os.path.join(path, "embeddings.pkl"), "rb") as f:
                self.embeddings = pickle.load(f)
        with open(os.path.join(path, "texts.pkl"), "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]

    def __len__(self) -> int:
        return len(self.texts)


class RetrieverAgent:
    """
    Retriever agent: dense retrieval over task-relevant corpora.

    Maintains separate indices for:
      - Web DOM/element snippets (Mind2Web)
      - Product descriptions (WebShop)
      - Tool documentation (AgentBench)
      - Episodic memory (past trajectories)
    """

    ROLE_NAME = "retriever"

    def __init__(
        self,
        backbone: BackboneManager,
        role_manager: RoleAdapterManager,
        comm_module: LatentCommunicationModule,
        agent_idx: int = 1,
        top_k: int = 5,
        embedding_pool: str = "mean",  # "mean" | "last" | "cls"
        confidence_threshold: float = 0.7,
    ):
        self.backbone = backbone
        self.role_manager = role_manager
        self.role_adapter = role_manager.get_adapter(self.ROLE_NAME)
        self.comm_module = comm_module
        self.agent_idx = agent_idx
        self.top_k = top_k
        self.embedding_pool = embedding_pool
        self.confidence_threshold = confidence_threshold

        hidden_size = backbone.hidden_size
        self.indices: Dict[str, FAISSMemoryIndex] = {
            "web_dom": FAISSMemoryIndex(hidden_size),
            "products": FAISSMemoryIndex(hidden_size),
            "tools": FAISSMemoryIndex(hidden_size),
            "episodic": FAISSMemoryIndex(hidden_size),
        }
        self.last_confidence: float = 1.0

    @torch.no_grad()
    def _encode_text(self, text: str, device: Optional[torch.device] = None) -> np.ndarray:
        """Encode text to a dense embedding using the backbone."""
        # Safety: ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text else "query"
        tokenizer = self.backbone.tokenizer
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        self.backbone.set_active_adapter(self.ROLE_NAME)
        outputs = self.backbone.model(
            **inputs,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
        mask = inputs["attention_mask"].unsqueeze(-1).float()

        if self.embedding_pool == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        elif self.embedding_pool == "last":
            lengths = mask.squeeze(-1).sum(dim=1).long()
            pooled = hidden[torch.arange(hidden.size(0)), lengths - 1, :]
        else:  # cls
            pooled = hidden[:, 0, :]

        return pooled[0].cpu().float().numpy()

    def index_passages(
        self,
        texts: List[str],
        index_name: str = "web_dom",
        metadata: Optional[List[Dict]] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ) -> None:
        """
        Encode and index a list of text passages.
        Uses batch encoding for efficiency.
        """
        if index_name not in self.indices:
            self.indices[index_name] = FAISSMemoryIndex(self.backbone.hidden_size)

        tokenizer = self.backbone.tokenizer
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            self.backbone.set_active_adapter(self.ROLE_NAME)
            with torch.no_grad():
                outputs = self.backbone.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(pooled.cpu().float().numpy())

            if (i // batch_size) % 10 == 0:
                logger.info(f"Indexed {min(i + batch_size, len(texts))}/{len(texts)} passages")

        embeddings = np.concatenate(all_embeddings, axis=0)
        self.indices[index_name].add(texts, embeddings, metadata)
        logger.info(f"Index '{index_name}' now has {len(self.indices[index_name])} passages")

    def retrieve(
        self,
        query: str,
        index_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[RetrievalResult], float]:
        """
        Retrieve top-K relevant passages for a query.

        Returns:
            results: List of RetrievalResult sorted by score
            confidence: Estimated confidence (top-1 score, normalized)
        """
        index_names = index_names or list(self.indices.keys())
        query_emb = self._encode_text(query, device=device)

        all_results: List[RetrievalResult] = []
        for name in index_names:
            idx = self.indices.get(name)
            if idx and len(idx) > 0:
                results = idx.search(query_emb, top_k=self.top_k)
                for r in results:
                    r.metadata["index"] = name
                all_results.extend(results)

        # Sort by score and take top-K
        all_results.sort(key=lambda r: r.score, reverse=True)
        top_results = all_results[: self.top_k]

        # Confidence: normalized top-1 score (IP in [-1,1] after normalization)
        confidence = (top_results[0].score + 1.0) / 2.0 if top_results else 0.0
        self.last_confidence = confidence

        return top_results, confidence

    def step(
        self,
        query: str,
        incoming_latent: Optional[torch.Tensor] = None,
        index_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, float]:
        """
        Retrieve and generate a retrieval response.

        Returns:
            action: Dict with retrieved passages
            hidden_states: For outgoing latent messages
            confidence: Retrieval confidence (for bitrate scheduler)
        """
        # Safety: ensure query is a string
        if not isinstance(query, str):
            query = str(query) if query else "query"
        results, confidence = self.retrieve(query, index_names, device)

        action = {
            "action": "retrieve",
            "query": query,
            "results": [
                {
                    "text": r.text[:200],
                    "score": r.score,
                    "source": r.metadata.get("index", "unknown"),
                }
                for r in results
            ],
            "confidence": confidence,
            "top_result": results[0].text[:500] if results else "",
        }

        # Get hidden states for outgoing latent comm
        hidden_states = self.backbone.get_hidden_states(
            *self.backbone.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).values(),
        )

        return action, hidden_states, confidence

    def add_episodic_memory(
        self,
        observation: str,
        action: str,
        reward: float,
        device: Optional[torch.device] = None,
    ) -> None:
        """Add a trajectory step to episodic memory."""
        text = f"[OBS]: {observation[:150]} [ACTION]: {action[:100]} [REWARD]: {reward:.2f}"
        emb = self._encode_text(text, device=device)
        self.indices["episodic"].add(
            [text],
            emb.reshape(1, -1),
            [{"reward": reward, "action": action}],
        )

    def generate_latent_message(
        self,
        hidden_states: torch.Tensor,
        k: int = 8,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.role_adapter.summarize_hidden_states(hidden_states, attention_mask, k=k)
