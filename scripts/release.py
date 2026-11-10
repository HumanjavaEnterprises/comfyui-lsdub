#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""release.py: the one release command. Maintainers only (it needs the private engine checkout and gh signed in).

  python scripts/release.py X.Y.Z [--dry-run] [--engine ../lsdub] [--site ../lsdub.web.microsite.src]

In order, stopping at the first failure with a plain message (nothing remote changes before step 9):
  1. refuses unless the engine and pack trees are clean, on main and equal to origin/main, and engine CI is green on its HEAD;
  2. refuses unless CHANGELOG.md has a `## X.Y.Z` section that passes release_notes.py, the microsite's main has a
     matching section in docs/changelog.html, and the demo zip fetch_demo.py points at is published with its checksum;
  3. bumps pyproject (version, Icon/Banner tag URLs) and CITATION.cff; regenerates the README node tables and the bug form;
  4. vendors the engine (a vendor record commit in the engine, pushed only in step 9);
  5. runs the pack checks CI runs (ruff registry rules, pytest, docs and models checks), including the private-methods scrub;
  6. commits "X.Y.Z: ..." on a release/X.Y.Z branch;
  7. pushes release/X.Y.Z and waits for CI and Install path to pass on that commit;
  8. (dry run stops before this: it works in a throwaway worktree and pushes nothing)
  9. fast-forwards main to it, tags vX.Y.Z, pushes main and the tag, pushes the engine record;
 10. watches the publish run, confirms the Comfy Registry lists X.Y.Z, then deploys the site from this box: a clean checkout of
     the microsite's main (equal to origin/main) through wrangler's own login (no API token, no GitHub secret), and checks
     https://lsdub.sndbnch.app/changelog answers 200 with the new version on it.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK_REPO = 'HumanjavaEnterprises/comfyui-lsdub'
ENGINE_REPO = 'HumanjavaEnterprises/lsdub'
SITE_REPO = 'HumanjavaEnterprises/lsdub.web.microsite.src'
REGISTRY = 'https://api.comfy.org/nodes/lsdub/versions'
EVIDENCE = ('Evidence: release-check source=lsdub/core/lyric_align.py read the vendored tree: vendor_engine scrub CHECK clean, '
            'pack tests and install-path green on release/{v}')


class Stop(SystemExit):
    def __init__(self, msg):
        pushed = '\n'.join(f'  - {p}' for p in PUSHED)
        tail = ('\nNothing remote was changed.' if not PUSHED else
                f'\nAlready pushed:\n{pushed}\nRecovery: CONTRIBUTING.md, Releases, "If release.py stops part way".')
        super().__init__(f'\nrelease: STOPPED. {msg}{tail}')


PUSHED = []
STEP = [0]


def step(msg):
    STEP[0] += 1
    print(f'\n[{STEP[0]}] {msg}', flush=True)


def sh(*cmd, cwd=PACK, check=True, quiet=False, env=None):
    if not quiet: print('    $', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise Stop(f'`{" ".join(cmd)}` failed (exit {r.returncode}) in {cwd}:\n{(r.stdout + r.stderr).strip()[-3000:]}')
    return r.stdout.strip()


def git(repo, *a, **k): return sh('git', *a, cwd=repo, **k)


def gh_json(*a): return json.loads(sh('gh', *a, quiet=True) or 'null')


def clean_main(repo, name):
    if git(repo, 'status', '--porcelain', quiet=True): raise Stop(f'the {name} tree at {repo} has uncommitted changes. Commit or stash them first.')
    br = git(repo, 'rev-parse', '--abbrev-ref', 'HEAD', quiet=True)
    if br != 'main': raise Stop(f'the {name} checkout is on {br!r}, not main.')
    git(repo, 'fetch', '-q', 'origin', 'main', quiet=True)
    if git(repo, 'rev-parse', 'HEAD', quiet=True) != git(repo, 'rev-parse', 'origin/main', quiet=True):
        raise Stop(f'the {name} main is not origin/main (pull or push first).')
    return git(repo, 'rev-parse', 'HEAD', quiet=True)


def runs_green(repo, sha, workflows, wait, timeout=4 * 3600):
    """every named workflow has a successful run on sha (waits for them when wait=True)."""
    t0 = time.time()
    while True:
        runs = gh_json('run', 'list', '-R', repo, '--commit', sha, '--json', 'workflowName,status,conclusion,url', '-L', '50')
        state = {}
        for w in workflows:
            mine = [r for r in runs if r['workflowName'] == w]
            if any(r['conclusion'] == 'success' for r in mine): state[w] = 'success'
            elif any(r['status'] != 'completed' for r in mine): state[w] = 'running'
            elif mine: state[w] = f"{mine[0]['conclusion']} ({mine[0]['url']})"
            else: state[w] = 'no run yet'
        bad = {w: s for w, s in state.items() if s not in ('success', 'running', 'no run yet')}
        if bad: raise Stop(f'{repo} @ {sha[:9]}: ' + '; '.join(f'{w} {s}' for w, s in bad.items()))
        if all(s == 'success' for s in state.values()): return
        if not wait: raise Stop(f'{repo} @ {sha[:9]} is not green yet: {state}')
        if time.time() - t0 > timeout: raise Stop(f'{repo} @ {sha[:9]}: gave up waiting: {state}')
        print(f'    waiting: {state}', flush=True)
        time.sleep(60)


def read(p): return open(p, encoding='utf-8').read()


def write(p, s): open(p, 'w', encoding='utf-8').write(s)


def bump(tree, v):
    pp = os.path.join(tree, 'pyproject.toml')
    s = read(pp)
    s = re.sub(r'^version = "[^"]+"', f'version = "{v}"', s, count=1, flags=re.M)
    s = re.sub(r'(comfyui-lsdub/)v\d+\.\d+\.\d+(/assets/)', rf'\g<1>v{v}\g<2>', s)
    write(pp, s)
    cf = os.path.join(tree, 'CITATION.cff')
    write(cf, re.sub(r'^version: "[^"]+"', f'version: "{v}"', read(cf), flags=re.M))


def demo_published(tree):
    src = read(os.path.join(tree, 'scripts', 'fetch_demo.py'))
    ver = re.search(r"^VERSION = '([^']+)'", src, re.M).group(1)
    tag = re.search(r"^TAG = (.+)$", src, re.M).group(1).strip()
    tag = f'v{ver}' if tag == "f'v{VERSION}'" else tag.strip("'")
    sha = re.search(r"^SHA256 = '([0-9a-f]{64})'", src, re.M).group(1)
    url = f'https://github.com/{PACK_REPO}/releases/download/{tag}/lsdub-demo-{ver}.zip'
    try:
        with urllib.request.urlopen(url, timeout=120) as r: got = hashlib.sha256(r.read()).hexdigest()  # noqa: S310 (fixed https URL)
    except OSError as e:
        got = f'unreachable ({e})'
    if got != sha:
        raise Stop(f'the demo zip fetch_demo.py points at is not published as pinned: {url} -> {got}, pinned {sha}.\n'
                   f'Attach it first (a new tag for a new zip, never a replaced asset):\n'
                   f'  gh release create {tag} dist/lsdub-demo-{ver}.zip -R {PACK_REPO} --latest=false '
                   f'--title "LSDub demo assets {ver}" --notes "Demo assets for the example workflows (scripts/fetch_demo.py)."')
    return url


def main():
    ap = argparse.ArgumentParser(description='LSDub: the one release command', epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('version')
    ap.add_argument('--dry-run', action='store_true', help='every check, in a throwaway worktree; pushes nothing')
    ap.add_argument('--engine', default=os.path.join(os.path.dirname(PACK), 'lsdub'))
    ap.add_argument('--site', default=os.path.join(os.path.dirname(PACK), 'lsdub.web.microsite.src'))
    a = ap.parse_args()
    v, dry = a.version, a.dry_run
    eng, site = os.path.abspath(a.engine), os.path.abspath(a.site)
    print(f'LSDub release {v}{" (DRY RUN: nothing is pushed)" if dry else ""}')
    if not re.fullmatch(r'\d+\.\d+\.\d+', v): raise Stop(f'{v!r} is not X.Y.Z')
    cur = re.search(r'^version = "([^"]+)"', read(os.path.join(PACK, 'pyproject.toml')), re.M).group(1)
    if tuple(map(int, v.split('.'))) <= tuple(map(int, cur.split('.'))): raise Stop(f'{v} is not newer than the current {cur}')
    for t in ('git', 'gh'):
        if not shutil.which(t): raise Stop(f'{t} is not on PATH')
    sh('gh', 'auth', 'status', quiet=True)

    step('engine and pack trees clean, on main, equal to origin/main; engine CI green on its HEAD')
    eng_sha = clean_main(eng, 'engine')
    pack_sha = clean_main(PACK, 'pack')
    runs_green(ENGINE_REPO, eng_sha, ['ci'], wait=False)
    if git(PACK, 'ls-remote', '--tags', 'origin', f'refs/tags/v{v}', quiet=True): raise Stop(f'tag v{v} already exists on origin')
    if git(PACK, 'ls-remote', '--heads', 'origin', f'release/{v}', quiet=True): raise Stop(f'branch release/{v} already exists on origin')
    print(f'    engine {eng_sha[:9]} (ci green), pack {pack_sha[:9]}')

    step(f'CHANGELOG section {v} (release_notes.py), the microsite changelog section, the demo zip')
    sys.path.insert(0, os.path.join(PACK, 'scripts'))
    import release_notes as RN
    try: RN.check(read(os.path.join(PACK, 'CHANGELOG.md')), v)
    except ValueError as e: raise Stop(f'CHANGELOG.md: {e}. Write the {v} section first (CONTRIBUTING.md, Release notes).') from None
    git(site, 'fetch', '-q', 'origin', 'main', quiet=True)
    r = subprocess.run(['git', 'show', 'origin/main:docs/changelog.html'], cwd=site, capture_output=True, text=True)
    if r.returncode: raise Stop(f'the microsite main ({site}) has no docs/changelog.html. The changelog page has to be on the site\'s main first.')
    page = r.stdout
    if f'id="v{v}"' not in page:
        raise Stop(f'the microsite main has no <section id="v{v}"> in docs/changelog.html. Add it on the site\'s main first.')
    print(f'    demo zip ok: {demo_published(PACK)}')

    tree, made_worktree = PACK, False
    if dry:
        tree = tempfile.mkdtemp(prefix=f'lsdub-release-{v}-')
        os.rmdir(tree)
        git(PACK, 'worktree', 'add', '-q', '--detach', tree, 'HEAD')
        made_worktree = True
    records = os.path.join(eng, 'scripts', 'vendor_records.tsv')
    records_before = read(records)
    try:
        step(f'bump to {v}: pyproject version + Icon/Banner URLs, CITATION.cff; regenerate README node tables and the bug form')
        if not dry: git(PACK, 'checkout', '-q', '-b', f'release/{v}')
        bump(tree, v)
        sh(sys.executable, 'scripts/make_readme_nodes.py', cwd=tree)

        step('vendor the engine')
        sh(sys.executable, os.path.join(eng, 'scripts', 'vendor_engine.py'), '--pack', tree, cwd=eng)

        step('pack checks (what CI runs, plus the private-methods scrub in tests/test_registry_rules.py)')
        sh(sys.executable, '-m', 'ruff', 'check', '--select', 'S102,S307,E702,F', '--target-version', 'py310', '.', cwd=tree)
        sh(sys.executable, '-m', 'pytest', '-q', 'tests', cwd=tree)
        sh(sys.executable, 'scripts/make_readme_nodes.py', '--check', cwd=tree)
        sh(sys.executable, 'scripts/check_models_md.py', cwd=tree)
        sh(sys.executable, 'scripts/release_notes.py', cwd=tree, quiet=True)

        step(f'commit {v} on release/{v}')
        title = RN.top_section(read(os.path.join(tree, 'CHANGELOG.md')))[1]
        first = next((ln[2:].strip() for ln in title if ln.startswith('- ')), '')
        git(tree, 'add', '-A')
        if dry:
            print(f'    would commit "{v}: {first[:100]}" with: ' + git(tree, 'diff', '--cached', '--stat', quiet=True).replace('\n', '\n      '))
            print('    would then run vendor_engine.py --check on the committed record')
            print(f'    would commit the engine vendor record ({EVIDENCE.format(v=v)[:60]}...)')
            print(f'\n[{STEP[0] + 1}..] would push release/{v}, wait for CI + Install path, fast-forward main, tag v{v}, push the tag and the '
                  'engine record, watch publish, confirm the Registry, check the site deploy.\nDRY RUN OK: every check passed.')
            return
        git(tree, 'commit', '-q', '-m', f'{v}: {first[:120]}')
        rel_sha = git(tree, 'rev-parse', 'HEAD', quiet=True)
        if read(records) != records_before:
            git(eng, 'commit', '-q', '-m', f'{v} vendor record\n\n{EVIDENCE.format(v=v)}', '--', 'scripts/vendor_records.tsv')
        sh(sys.executable, os.path.join(eng, 'scripts', 'vendor_engine.py'), '--pack', tree, '--check', cwd=eng)   # clean engine again

        step(f'push release/{v} and wait for CI + Install path on {rel_sha[:9]}')
        git(PACK, 'push', '-q', 'origin', f'release/{v}')
        PUSHED.append(f'pack branch release/{v} ({rel_sha[:9]}); main, tags and the Registry untouched')
        print('    (a branch only; main, tags and the Registry are untouched until both pass)')
        time.sleep(20)
        runs_green(PACK_REPO, rel_sha, ['CI', 'Install path'], wait=True)

        step(f'fast-forward main, tag v{v}, push; push the engine record')
        git(PACK, 'checkout', '-q', 'main')
        git(PACK, 'merge', '-q', '--ff-only', f'release/{v}')
        git(PACK, 'tag', '-a', f'v{v}', '-m', f'LSDub {v}')
        git(PACK, 'push', '-q', '--atomic', 'origin', 'main', f'v{v}')   # main and the tag land together or not at all
        PUSHED.append(f'pack main -> {rel_sha[:9]} and tag v{v} (publish starts from the tag)')
        git(eng, 'push', '-q', 'origin', 'main')
        PUSHED.append('engine main: the vendor record commit')

        step('watch publish (it re-runs Install path on the tag first)')
        time.sleep(20)
        runs_green(PACK_REPO, rel_sha, ['Publish to the Comfy Registry'], wait=True)

        step(f'the Comfy Registry lists {v}')
        for _ in range(30):
            with urllib.request.urlopen(REGISTRY, timeout=60) as r: vs = {x.get('version'): x.get('status') for x in json.loads(r.read())}  # noqa: S310
            if v in vs:
                st = {'NodeVersionStatusActive': 'Active', 'NodeVersionStatusPending': 'Pending'}.get(vs[v], vs[v])
                print(f'    {v}: {st}' + (' (listed; the Registry still has to review it before it is Active)' if st == 'Pending' else ''))
                break
            time.sleep(30)
        else:
            raise Stop(f'publish passed but {REGISTRY} does not list {v} after 15 minutes; check the publisher page.')

        step('site deploy from this box (wrangler login; clean microsite main), then /changelog shows the version')
        deploy_site(site, v)
        print(f'\nreleased {v}.')
    finally:
        if made_worktree:
            git(PACK, 'worktree', 'remove', '--force', tree, check=False, quiet=True)
            if read(records) != records_before: write(records, records_before)   # the dry run's vendor record, undone


CF_ACCOUNT = 'fa095a45453409a42f1c739a174e609f'
CHANGELOG_URL = 'https://lsdub.sndbnch.app/changelog'


def deploy_site(site, v):
    """the microsite deploys from a clean checkout of its main, equal to origin/main, with wrangler's OAuth login on this box (an
    API token in the environment is dropped so the login is what deploys); then /changelog must answer 200 and carry v."""
    git(site, 'fetch', '-q', 'origin', 'main', quiet=True)
    if git(site, 'rev-parse', '--abbrev-ref', 'HEAD', quiet=True) != 'main':
        raise Stop(f'the microsite checkout ({site}) is not on main. Switch it to main and deploy by hand: CONTRIBUTING.md, Releases.')
    if git(site, 'status', '--porcelain', quiet=True):
        raise Stop(f'the microsite checkout ({site}) has local changes. Deploy needs a clean main.')
    if git(site, 'rev-parse', 'HEAD', quiet=True) != git(site, 'rev-parse', 'origin/main', quiet=True):
        raise Stop(f'the microsite main ({site}) is not equal to origin/main. Pull it, then deploy by hand: CONTRIBUTING.md, Releases.')
    env = {k: val for k, val in os.environ.items() if k != 'CLOUDFLARE_API_TOKEN'}
    env['CLOUDFLARE_ACCOUNT_ID'] = CF_ACCOUNT
    sh('npx', 'wrangler', 'deploy', cwd=site, env=env)
    PUSHED.append('the microsite deployed from its main')
    for _ in range(10):
        try:
            with urllib.request.urlopen(CHANGELOG_URL, timeout=30) as r:  # noqa: S310
                if r.status == 200 and f'id="v{v}"' in r.read().decode('utf-8', 'replace'):
                    print(f'    {CHANGELOG_URL}: 200, shows {v}')
                    return
        except OSError:
            pass
        time.sleep(15)
    raise Stop(f'the site deployed but {CHANGELOG_URL} does not answer 200 with {v} on it; check the deploy output above.')


if __name__ == '__main__':
    main()
