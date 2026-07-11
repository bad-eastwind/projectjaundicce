"""Domain-generalization classification objectives (baselines + comparison):
   erm      - plain cross-entropy
   irm      - Invariant Risk Minimization (Arjovsky et al.) penalty across environments (domains)
   groupdro - Group Distributionally Robust Optimization (Sagawa et al.), upweights worst domain

Environments/groups = the discovered pseudo-domains (manifest `domain` column). These are the DG
baselines our physics+causal method is compared against; they replace only the classification term,
so they still compose with any aux losses.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


class DGObjective:
    def __init__(self, method: str = "erm", num_groups: int = 1,
                 lam: float = 1.0, eta: float = 0.01, anneal_steps: int = 0):
        self.method = method
        self.num_groups = max(int(num_groups), 1)
        self.lam = lam
        self.eta = eta
        self.anneal_steps = anneal_steps
        self.step = 0
        self.q = torch.ones(self.num_groups) / self.num_groups   # GroupDRO group weights

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.ones(1, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits * scale, y)
        g = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return (g ** 2).sum()

    def __call__(self, logits, y, domain):
        self.step += 1
        if self.method == "erm" or domain is None:
            return F.cross_entropy(logits, y), {}

        groups = domain.unique()
        if self.method == "irm":
            ce = pen = 0.0
            for gd in groups:
                m = domain == gd
                ce = ce + F.cross_entropy(logits[m], y[m])
                pen = pen + self._irm_penalty(logits[m], y[m])
            n = len(groups)
            ce, pen = ce / n, pen / n
            lam = self.lam if (self.anneal_steps == 0 or self.step > self.anneal_steps) else 1.0
            return ce + lam * pen, {"irm_pen": float(pen.item())}

        if self.method == "groupdro":
            self.q = self.q.to(logits.device)
            losses, idxs = [], []
            for gd in groups:
                m = domain == gd
                losses.append(F.cross_entropy(logits[m], y[m])); idxs.append(int(gd))
            with torch.no_grad():
                for l, gi in zip(losses, idxs):
                    self.q[gi] = self.q[gi] * torch.exp(self.eta * l.detach())
                self.q = self.q / self.q.sum()
            loss = sum(self.q[gi] * l for l, gi in zip(losses, idxs))
            return loss, {"worst_group": float(max(l.item() for l in losses))}

        return F.cross_entropy(logits, y), {}
