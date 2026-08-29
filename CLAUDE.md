# Working conventions for this repo

## Never reference `Planning/` in shipped code, docs, or PRs

`Planning/` (the spec, `DAY1.md`/`DAY2.md`/`DAY3.md`, the eval-failure and manual-notes fix
plans, etc.) holds internal working/build notes for this project — not something a reader of the
code, `README.md`, `EVAL_SCHEMA.md`, `EDGE_CASES.md`, or a PR description should ever see cited.
This applies to:

- Code comments and docstrings (e.g. no `# see Planning/DAY3.md §2.2`, no `(the spec's §8.1)`).
- `README.md`, `EVAL_SCHEMA.md`, `EDGE_CASES.md`, and any other doc committed to the repo.
- PR titles and PR descriptions (`gh pr create` / `gh pr edit`) — not just the diff. If a PR
  description was drafted by lifting language from a plan doc's section headers ("DAY3 §2",
  "Group 7", etc.), rewrite it in the project's own words before opening the PR.
- Branch names and commit messages, as a lower but still real priority — prefer a name/message
  that describes the change itself over one that echoes a plan doc's own section numbering.

If a design decision genuinely traces back to something written in `Planning/`, restate the
*reasoning* in the artifact's own words instead of citing the document by name or section number.
A reader of this repo should never need to open `Planning/` to understand why the code or the
docs say what they say.

This has needed a cleanup pass more than once — check for it explicitly (grep for
`Planning/`, `DAY1`/`DAY2`/`DAY3`, `spec`, `§`) before opening a PR, not just at the end of a
multi-day session.
