# Contributing to LSDub

Thanks for helping. LSDub gets better every time someone throws a real clip at it.

## Report a bug or a bad shot
Open an issue with the **Bug report** form. The most useful thing you can attach is the **Verify** or **Fix-It**
output for the shot, plus a short clip or a frame grab. Only share media you have the rights to share.

## Suggest something
Use the **Suggestion** form. Tell us the shot or the music video that made you want it.

## Pull requests
- Small and focused beats big and sweeping.
- Run the checks before you open it:
  - `pip install -r requirements.txt pytest ruff==0.17.3 comfy-cli packaging` (what CI installs; the deploy-inputs test runs comfy-cli's own scanner and needs it)
  - `ruff check --select S102,S307,E702,F .`
  - `pytest -q`
- New nodes: V3 schema, a tooltip on every input and output.
- Keep it open: open-weights models only, no partner or API nodes in the core path, no code you can't license
  under Apache-2.0.
- Credit what you built on, in the README's "Built on and learned from", if your change uses someone's work directly.

By contributing you agree your work is licensed under the Apache License 2.0, the same as the rest of the pack.

## Release notes
- Each version's section in CHANGELOG.md is what the Comfy Registry shows: at most 6 short topline bullets in
  plain words (what a ComfyUI user gets), then a last line `Full changelog: https://lsdub.sndbnch.app/changelog`.
  The full detail of every change goes on that page, not here.
- No `## Unreleased` heading in the file; add the section when the version is released.
- The heading names the version in pyproject.toml. `python scripts/release_notes.py` checks all of this; CI runs it
  (`tests/test_release_notes.py`) and the publish job fails without it.
