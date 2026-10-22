"""Shared TSV/Liberty parsing for the sizing tool.

No experiment runner, physical evaluation or model state lives here.
"""
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluator'))
from internal import legality

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


@dataclass
class DesignView:
    """One parsed snapshot, including the public protected-instance policy."""
    cells: dict
    pins: list
    protected: set
    views: dict

    @classmethod
    def load(cls, prefix, library, views):
        # Reuse public parsing and input validation, not a parallel policy.
        design = legality.Design(prefix, library)
        pins = [pin for instance in design.pins.values() for pin in instance.values()]
        return cls(design.cells, pins, design.protected, views)


@dataclass
class TimingData:
    """Native path rows (ps) and arc reports (native ps/fF lookup inputs)."""
    paths: list
    arcs: list
    folder: Path

    def read_report(self, sample):
        """Read only reports requested by the model's inclusion traversal."""
        return (self.folder/sample['report_file']).read_text()


def load_timing_data(folder):
    folder = Path(folder)
    return TimingData(rows(folder/'paths.tsv'), rows(folder/'arcs.tsv'), folder)


def load_analysis_inputs(folder, catalog):
    folder = Path(folder)
    views = {}
    for row in rows(folder/'views.tsv'):
        views.setdefault(row['master'], set()).add(row['view'])
    design = DesignView.load(folder/'snapshot', catalog.library,
                             {k: sorted(v) for k, v in views.items()})
    return design, load_timing_data(folder)


def groups(text, kind):
    """Read balanced named Liberty groups, skipping braces inside strings/comments."""
    start = re.compile(r'\b' + re.escape(kind) + r'\s*\(([^)]*)\)\s*\{')
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*|[{}]', re.S)
    pos = 0
    while match := start.search(text, pos):
        depth = 1
        for token in tokens.finditer(text, match.end()):
            if token[0] == '{':
                depth += 1
            elif token[0] == '}':
                depth -= 1
                if depth == 0:
                    yield match[1].strip().strip('"'), text[match.end():token.start()]
                    pos = token.end()
                    break
        else:
            raise ValueError('Unbalanced Liberty group ' + kind)


def attribute(text, name, default=None):
    value = re.search(r'\b' + re.escape(name) + r'\s*:\s*("[^"]*"|[^;]+);', text)
    return value[1].strip().strip('"') if value else default


def condition(text):
    # Only simple conjunctions are canonicalized; never erase arbitrary grouping.
    clean = re.sub(r'\s+', '', text).replace('*', '&')
    if not clean:
        return ''
    if '!(' in clean or clean.count('(') != clean.count(')'):
        raise ValueError('Unsupported grouped negation or parentheses: ' + text)
    unwrapped = clean.replace('(', '').replace(')', '')
    if not re.fullmatch(r'!?\w+(?:&!?\w+)*', unwrapped) or '+' in clean or '|' in clean:
        raise ValueError('Unsupported conditional expression: ' + text)
    return '&'.join(sorted(unwrapped.split('&')))
