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
  - `pip install -r requirements.txt pytest ruff`
  - `ruff check --select S102,S307,E702,F .`
  - `pytest -q`
- New nodes: V3 schema, a tooltip on every input and output.
- Keep it open: open-weights models only, no partner or API nodes in the core path, no code you can't license
  under Apache-2.0.
- Credit what you built on, in the README's "Built on and learned from", if your change uses someone's work directly.

By contributing you agree your work is licensed under the Apache License 2.0, the same as the rest of the pack.
