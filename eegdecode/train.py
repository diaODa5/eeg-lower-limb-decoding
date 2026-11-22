"""Train causal K0 teachers, cache source targets, and fit the streaming student."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import labels_and_stability, normalize_source, source_trials
from .distillation import mild_view, student_loss, teacher_targets
from .model import Hybrid, detach_state


@torch.no_grad()
def replay(model, x, device, chunk=4000):
    model.eval()
    state = None
    scores, endpoints = [], []
    for start in range(0, x.shape[-1], chunk):
        part = torch.from_numpy(x[None, :, start:start+chunk]).to(device)
        logits, ends, state = model.forward_chunk(part, state)
        scores.append(logits[0].cpu().numpy())
        endpoints.append(ends.cpu().numpy())
    return np.concatenate(endpoints), np.concatenate(scores)


def build_targets(teachers, records, device):
    cache = {}
    for trial, record in records.items():
        x = record["x"]
        view = mild_view(x, trial)
        expected = np.arange(1000, x.shape[-1] + 1, 125)
        clean_logits, view_logits = [], []
        for teacher in teachers:
            ends, clean = replay(teacher, x, device)
            noisy_ends, noisy = replay(teacher, view, device)
            if not np.array_equal(ends, expected) or not np.array_equal(noisy_ends, expected):
                raise ValueError(f"Unaligned teacher endpoints in {trial}")
            clean_logits.append(clean)
            view_logits.append(noisy)
        cache[trial] = teacher_targets(clean_logits, view_logits)
    return cache


def fit(records, device, seed, epochs, lr, max_entry, cache=None):
    torch.manual_seed(seed)
    model = Hybrid().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    counts = sum(np.bincount(r["labels"][r["stable"]], minlength=4) for r in records.values())
    if np.any(counts == 0):
        raise ValueError("All four classes must have stable source samples")
    weights = torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        order = rng.permutation(list(records)).tolist()
        seen = 0
        for offset in range(0, len(order), 4):
            ids = order[offset:offset+4]
            entry = int(rng.integers(max_entry // 125 + 1)) * 125
            length = min(records[t]["x"].shape[-1] for t in ids)
            state = None
            for start in range(entry, length, 4000):
                x = np.stack([records[t]["x"][:, start:start+4000] for t in ids])
                logits, ends, state = model.forward_chunk(torch.from_numpy(x).to(device), state)
                relative = ends.cpu().numpy()
                absolute = entry + relative
                keep = np.stack([records[t]["stable"][absolute-1] for t in ids]) & (relative >= 2000)
                if keep.any():
                    mask = torch.from_numpy(keep).to(device)
                    targets = np.stack([records[t]["labels"][absolute-1] for t in ids])[keep]
                    labels = torch.from_numpy(targets).to(device)
                    if cache is None:
                        loss = F.cross_entropy(logits[mask], labels, weight=weights)
                    else:
                        ix = (absolute - 1000) // 125
                        soft = np.stack([cache[t][0][ix] for t in ids])[keep]
                        reliability = np.stack([cache[t][1][ix] for t in ids])[keep]
                        loss = student_loss(logits[mask], labels, weights,
                                            torch.from_numpy(soft).to(device),
                                            torch.from_numpy(reliability).to(device))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    seen += len(labels)
                state = detach_state(state)
        if not seen:
            raise ValueError("No eligible stable training decisions after burn-in")
        print(f"seed={seed} epoch={epoch} stable_decisions={seen}", flush=True)
    return model.eval()


def synthetic_records():
    rng = np.random.default_rng(42)
    codes = np.repeat([402, 404], [2125, 1375])
    records = {}
    for state in ("static", "motion"):
        labels, stable = labels_and_stability(codes, state, 1000)
        x = rng.normal(size=(28, len(codes))).astype(np.float32)
        records[f"synthetic_{state}"] = dict(x=x, labels=labels, stable=stable)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Small synthetic end-to-end run")
    source = commands.add_parser("source", help="Fit one subject using frozen source budgets")
    source.add_argument("--archive", required=True, type=Path, help="Figshare Opened_loop.zip")
    source.add_argument("--subject", choices=[f"A{i:02d}" for i in range(1, 6)], required=True)
    for role in ("teacher", "student"):
        source.add_argument(f"--{role}-epochs", type=int, required=True)
        source.add_argument(f"--{role}-lr", type=float, required=True)
    source.add_argument("--seed", type=int, default=42)
    smoke.set_defaults(teacher_epochs=1, student_epochs=1, teacher_lr=.001, student_lr=.001, seed=42)
    for command in (source, smoke):
        command.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
        command.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if min(args.teacher_epochs, args.student_epochs) < 1 or not all(
        np.isfinite(x) and x > 0 for x in (args.teacher_lr, args.student_lr)
    ):
        parser.error("Each training budget needs positive epochs and a positive finite learning rate")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    synthetic = args.command == "smoke"
    records = synthetic_records() if synthetic else source_trials(args.archive, args.subject)
    norm = normalize_source(records)
    args.out.mkdir(parents=True, exist_ok=False)
    max_entry = 0 if synthetic else 10000
    teachers = [fit(records, device, seed, args.teacher_epochs, args.teacher_lr, max_entry)
                for seed in (9001, 9002, 9003)]
    cache = build_targets(teachers, records, device)
    student = fit(records, device, args.seed, args.student_epochs, args.student_lr, max_entry, cache)
    r = np.concatenate([targets[1] for targets in cache.values()])
    checkpoint = dict(arm="A0_full_L10_safe", weights=student.cpu().state_dict(),
                      source_normalization=norm, synthetic=synthetic, formal_result=False,
                      teacher_seeds=[9001, 9002, 9003], student_seed=args.seed,
                      teacher_epochs=args.teacher_epochs, teacher_lr=args.teacher_lr,
                      student_epochs=args.student_epochs, student_lr=args.student_lr)
    tmp = args.out / "student.pt.tmp"
    torch.save(checkpoint, tmp)
    tmp.replace(args.out / "student.pt")
    print(f"source_endpoints={len(r)} mean_reliability={r.mean():.6f} "
          f"synthetic={synthetic} saved={args.out / 'student.pt'}")


if __name__ == "__main__":
    main()
