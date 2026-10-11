"""Fit the two-stage draft head basis: top-r principal directions of captured drafter hidden states.
usage: fit_draft_basis.py HIDDEN.pt[,HIDDEN2.pt...] OUT.pt [r=256]"""
import sys, torch
H = torch.cat([torch.load(p).float() for p in sys.argv[1].split(",")])
r = int(sys.argv[3]) if len(sys.argv) > 3 else 256
ev, vec = torch.linalg.eigh(H.T @ H)
P = vec.flip(1)[:, :r].contiguous()
print(f"{H.shape[0]} states, rank {r} captures {float(ev.flip(0)[:r].sum() / ev.sum()):.4f} of the energy")
torch.save({"P": P, "n": H.shape[0]}, sys.argv[2])
