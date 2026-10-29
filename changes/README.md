# News fragments

Each pull request into `development` adds a news fragment here: one Markdown file that tells users,
in a sentence or two, what changed for them. towncrier collects the fragments into
[`CHANGELOG.md`](../CHANGELOG.md) when a release is made, and deletes them.

Name the file `<issue>.<type>.md`, e.g. `102.bugfix.md`, or `+<slug>.<type>.md` for a change with
no issue. The types are:

| Type | For |
| --- | --- |
| `feature` | New behaviour or options |
| `bugfix` | Fixes, above all ones that change a run's results |
| `removal` | Removed or deprecated API, and changed defaults |
| `doc` | Documentation |
| `misc` | Packaging, dependencies, CI |

```bash
towncrier create 102.bugfix.md --content "Combined-spw solutions are applied when the selection's spw is empty."
towncrier build --draft --version 1.1.0   # preview the changelog
```

A pull request that users won't notice, e.g. a refactor or a test, can skip the fragment with the
`no-changelog` label. Dependabot's pull requests skip it too.
