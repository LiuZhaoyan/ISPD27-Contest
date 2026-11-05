"""Preserve fixed input DEF grid declarations after OpenROAD routing export.

The router's generated GCell grid belongs to the evaluation flow. Submissions
must retain the input's fixed DEF data; the organizer checks this before routing.
"""
import argparse
import re
from pathlib import Path

GRID = re.compile(r'^\s*GCELLGRID\s+[^;]*;[^\S\n]*\n?', re.MULTILINE)


def preserve_input_grid(source, output):
    original = Path(source).read_text()
    exported = Path(output).read_text()
    declarations = ''.join(m.group(0).lstrip('\n') for m in GRID.finditer(original))
    exported = GRID.sub('', exported)
    if declarations:
        anchor = re.search(r'^COMPONENTS\s', exported, re.MULTILINE)
        if anchor is None:
            raise ValueError('Missing DEF COMPONENTS section')
        exported = exported[:anchor.start()] + declarations + '\n' + exported[anchor.start():]
    Path(output).write_text(exported)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    preserve_input_grid(args.source, args.output)
