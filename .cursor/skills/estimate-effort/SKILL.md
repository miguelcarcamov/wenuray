---
name: estimate-effort
description: >-
  Estimate the effort of a Wenuray GitHub issue and set its effort:N label (Fibonacci). Use when an
  issue has no effort label, or when asked to size or prioritise issues.
---

# Effort estimation

## Input

The whole issue: description, acceptance criteria, open questions, labels, milestone, linked
issues and PRs. Check the code it touches.

## Output

1. One `effort:N` label, N in 1, 2, 3, 5, 8, 13 (`gh issue edit N --add-label effort:3`; remove an
   old one).
2. One comment: the effort, 2–4 sentences of rationale (modules touched, test surface, CASA or
   Pyralysis involvement, risk), and the tests it needs.

## Rubric

| Effort | Typical scope |
| --- | --- |
| 1 | Docs, a one-line fix, a trivial test |
| 2 | A small change in one or two files, unit tests |
| 3 | A feature across a few modules, unit tests, maybe one integration test |
| 5 | Cross-module change, integration tests on simulated data, docs and recipes |
| 8 | A new backend or calibration type, the loop, or upstream coordination (CASA, Pyralysis) |
| 13 | Architecture: split it into issues of 8 or less instead |

Raise the estimate when the change needs a design decision (`design` label), touches the
self-calibration invariants, or depends on an unreleased upstream fix.
