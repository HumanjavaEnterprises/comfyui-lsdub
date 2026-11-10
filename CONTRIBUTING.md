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

## Stability policy
- Releases are batched; patch versions only for breakage. Features go into the next minor version.
- Defaults stay unchanged within a minor series unless the changelog says otherwise.
- Every release passes the install-path test (`.github/workflows/install-path.yml`, `scripts/install_path_check.py`) on a
  clean ComfyUI before it is tagged. Run it yourself: `python scripts/install_path_check.py --comfyui PATH/TO/ComfyUI` with
  that ComfyUI's venv (its requirements plus `requirements.txt`, nothing else).
- Models are pinned by revision and sha256.
- 0.2.0 to 0.2.2 landed in one day while the install-path test did not yet exist; it runs on every release from 0.2.3 on.

## Releases (maintainers)
One command, from a clean pack checkout on main with the engine and the microsite checked out beside it:

    python scripts/release.py X.Y.Z --dry-run     # every check, in a throwaway worktree; pushes nothing
    python scripts/release.py X.Y.Z

Before it: write the `## X.Y.Z` section in CHANGELOG.md (see Release notes) and the matching `<section id="vX.Y.Z">` in the
microsite's `docs/changelog.html` on its main. The command refuses unless both trees are clean and on main and engine CI is
green; bumps the version, the Icon/Banner URLs and CITATION.cff; regenerates the node tables and the bug form; vendors the
engine; runs the pack checks; pushes `release/X.Y.Z` and waits for CI and Install path; then fast-forwards main, tags `vX.Y.Z`,
watches the publish (which runs Install path again first), confirms the Registry lists the version and checks the site
deploy. Any failure before the tag stops with a plain message and leaves nothing changed remotely.

If release.py stops part way:
- Before "push release/X.Y.Z": nothing remote changed. Fix the cause; delete the local `release/X.Y.Z` branch
  (`git checkout main && git branch -D release/X.Y.Z`) and drop a local engine record commit if one was made
  (`git -C ../lsdub reset --hard origin/main`), then run it again.
- After the release branch is pushed but before the tag: only `release/X.Y.Z` is on origin. Fix on that branch and push
  again (release.py waits on the new commit), or delete it (`git push origin --delete release/X.Y.Z`) and start over.
- After the tag: main and `vX.Y.Z` were pushed together (`--atomic`), so publish is running. Never re-tag or delete the tag;
  watch the publish run, and if it failed, fix and release the next patch version. Push the engine record by hand if
  release.py stopped before it did (`git -C ../lsdub push origin main`).
release.py names what it already pushed whenever it stops.

Repository secrets:
- `REGISTRY_ACCESS_TOKEN` (here): the Comfy Registry publisher key, already set.
- `MICROSITE_DISPATCH_TOKEN` (here): a fine-grained GitHub token, resource owner HumanjavaEnterprises, repository access only
  `lsdub.web.microsite.src`, permission Contents: Read and write (what `POST /repos/.../dispatches` needs). Without it the
  publish job's site step prints a notice and skips, and `release.py` starts the deploy itself with your `gh` login.
- `CLOUDFLARE_API_TOKEN` (in the microsite repo): see its README, Deploys.
