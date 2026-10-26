---
name: pr-review
description: >-
  Shallow review of an open Wenuray pull request: check CI, fix failures and review comments within
  the PR's scope, resolve mechanical conflicts, and escalate to the maintainer when a fix needs a
  design decision. Use on an open PR ("use pr-review for PR #N"). For a deep review of a diff
  against the rules, use convention-review.
---

# Pull request review

Bring one PR toward mergeable with **shallow** fixes only. Don't redesign in this pass.

## Before acting

1. `gh pr view N --comments` and the linked issue. `gh pr checkout N` for the branch.
2. Read `AGENTS.md` and `.cursor/rules/`.

## 1. CI first

1. `gh pr checks N`. Green: go to review threads.
2. Failed: read the first failing job's log (`gh run view <run> --log-failed`) and find the first
   actionable error.
3. Fix only failures caused by this PR, within the shallow scope below. Never weaken CI, skip or
   loosen tests, or change unrelated workflows to get green.

## 2. Review threads

1. List unresolved threads.
2. Clear, shallow requests: fix, push, reply with one line saying what changed.
3. Wrong or out-of-scope requests: reply briefly why; don't implement.

## 3. Conflicts

Merge `origin/development` into the branch only when resolving is mechanical (same intent on both
sides). A conflict that needs a design choice: stop and escalate.

## 4. Shallow scope

Typos, lint, a missing import, a test assertion tied to the diff, a small local fix that CI or a
comment clearly requires, a mechanical conflict. Smallest correct diff, within the PR's purpose.

## 5. Escalate

When fixes would change the architecture, public API or conventions in different ways, need a
larger refactor, the CI failure's cause is unclear or outside the PR, or feedback is ambiguous,
comment on the PR:

```text
**Maintainer decision needed**

<what failed or was asked, and why an automatic fix is unsafe>

Options: <the choices, with a recommendation>
```

## 6. Gates and handoff

```bash
pre-commit run --files <changed files>
pytest <targeted tests>
```

Push to the PR branch and leave it open: the maintainer merges.
