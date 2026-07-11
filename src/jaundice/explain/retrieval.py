"""Retrieval explanation head (case-based explainability).

Post-hoc, non-parametric layer over the trained model's disentangled embedding (the head-input vector
= concat(pooled, bilirubin feature)). We build a reference bank of L2-normalized embeddings from the
training split, then for any query image retrieve its k nearest labeled exemplars (cosine). This gives:
  - a case-based explanation: "most similar jaundice / normal cases" (the neighbor images),
  - a non-parametric prediction: distance-weighted vote over neighbor labels,
  - an uncertainty: entropy of that vote (low neighbor agreement -> high uncertainty).
No training; runs after the classifier is trained. Complements the physical BiliAxis heatmap.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


class RetrievalBank:
    def __init__(self, embeddings, labels, paths, domains):
        self.emb = F.normalize(embeddings, dim=1)     # [M,E]
        self.labels = labels                          # [M]
        self.paths = list(paths)                      # M
        self.domains = domains                        # [M]

    @classmethod
    @torch.no_grad()
    def build(cls, model, loader, dev):
        model.eval()
        embs, labs, paths, doms = [], [], [], []
        for b in loader:
            _, aux = model(b["image"].to(dev), return_aux=True)
            embs.append(aux["embedding"].float().cpu())
            labs.append(b["label"]); paths += list(b["path"]); doms.append(b["domain"])
        return cls(torch.cat(embs), torch.cat(labs), paths, torch.cat(doms))

    @torch.no_grad()
    def query(self, emb: torch.Tensor, k: int = 5, tau: float = 0.1):
        """Returns (retrieval_prob_jaundice [Q], uncertainty [Q], neighbor_idx [Q,k], sims [Q,k])."""
        q = F.normalize(emb.float().cpu(), dim=1)
        k = min(k, self.emb.shape[0])
        sims = q @ self.emb.t()                        # [Q,M] cosine
        vals, idx = sims.topk(k, dim=1)
        w = torch.softmax(vals / tau, dim=1)           # distance-weighted
        nl = self.labels[idx].float()                  # [Q,k]
        prob = (w * nl).sum(1).clamp(1e-6, 1 - 1e-6)   # P(jaundice)
        ent = -(prob * prob.log() + (1 - prob) * (1 - prob).log())   # binary entropy
        return prob, ent, idx, vals

    def save(self, path):
        torch.save({"emb": self.emb, "labels": self.labels,
                    "paths": self.paths, "domains": self.domains}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=True)  # bank = tensors + primitives
        b = cls.__new__(cls)
        b.emb, b.labels, b.paths, b.domains = d["emb"], d["labels"], d["paths"], d["domains"]
        return b
