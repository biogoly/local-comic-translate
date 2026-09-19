"""Check extracted UI coverage, runtime contexts, and compiled Qt catalogs.

Run with the app's Python environment:
    python resources/translations/check_translations.py
Add --compile to rebuild the checked-in .qm files after editing .ts files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import string
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import PySide6
from PySide6.QtCore import QCoreApplication, QTranslator

CATALOG_DIR = Path(__file__).resolve().parent
ROOT = CATALOG_DIR.parent.parent
LANGUAGES = ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN')
# QObject.tr() uses the concrete QObject class, including calls from mixins.
RUNTIME_CONTEXTS = {
    'WorkspaceMixin': ('ComicTranslate', 'ComicTranslateUI'),
    'self.main': ('ComicTranslate', 'ComicTranslateUI'),
    'self.settings.ui': ('SettingsPageUI',),
}


def entries(tree):
    return {(c.findtext('name'), m.findtext('source')): m
            for c in tree.findall('context') for m in c.findall('message')}


def placeholders(text):
    fields = [(name, spec, conversion) for _, name, spec, conversion
              in string.Formatter().parse(text) if name is not None]
    return Counter(fields), Counter(re.findall(r'%[1-9]\d*|%n', text))


def qt_tool(name):
    folder = Path(PySide6.__file__).resolve().parent
    for candidate in (folder / (name + '.exe'), folder / name, folder / 'Qt/libexec' / name,
                      folder / 'Qt/bin' / name):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f'Cannot find Qt {name} in {folder}')


def check(compile_catalogs=False):
    app = QCoreApplication.instance() or QCoreApplication([])
    # Repeating an absolute checkout path for every source can exceed Windows'
    # command-line limit, especially for long user/worktree directory names.
    sources = sorted(str(p.relative_to(ROOT)) for d in ('app', 'modules', 'pipeline')
                     for p in (ROOT / d).rglob('*.py'))
    sources += ['comic.py', 'controller.py']
    with tempfile.TemporaryDirectory(prefix='comic-i18n-') as tmp:
        inventory = Path(tmp) / 'inventory.ts'
        # Pass files explicitly: some Qt builds don't recurse Python directories.
        subprocess.run([qt_tool('lupdate'), *sources, '-no-obsolete', '-ts', str(inventory)],
                       check=True, cwd=ROOT)
        required = entries(ET.parse(inventory))
    for (context, source), message in list(required.items()):
        for runtime in RUNTIME_CONTEXTS.get(context, ()):
            required.setdefault((runtime, source), message)
    failures = []
    for language in LANGUAGES:
        path = CATALOG_DIR / f'ct_{language}.ts'
        catalog = entries(ET.parse(path))
        qm = CATALOG_DIR / 'compiled' / f'ct_{language}.qm'
        if compile_catalogs:
            subprocess.run([qt_tool('lrelease'), str(path), '-qm', str(qm)], check=True)
        translator = QTranslator()
        if not translator.load(str(qm)):
            failures.append(f'{language}: cannot load {qm.name}')
            continue
        for (context, source) in required:
            message = catalog.get((context, source))
            translation = message.find('translation') if message is not None else None
            if translation is None or not translation.text or translation.get('type') in {'unfinished', 'vanished', 'obsolete'}:
                failures.append(f'{language}: missing {context}: {source!r}')
                continue
            value = translation.text
            try:
                if placeholders(source) != placeholders(value):
                    failures.append(f'{language}: placeholder mismatch {context}: {source!r} -> {value!r}')
            except ValueError:
                failures.append(f'{language}: invalid format fields {context}: {source!r}')
            if translator.translate(context, source) != value:
                failures.append(f'{language}: stale/missing compiled entry {context}: {source!r}')
        print(f'{language}: checked {len(required)} active/runtime messages')
    if failures:
        raise SystemExit('\n'.join(failures))
    print('All catalogs cover the active UI, preserve placeholders, and match compiled resources.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true')
    check(parser.parse_args().compile)
