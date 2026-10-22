"""Nearest legal sizing actions, backed by the official catalog and snapshots."""
import csv
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

from data_io import ROOT, legality

VT = {'SL': 0, 'L': 1, 'R': 2}
DIRECTIONS = ('drive_down', 'drive_up', 'vt_down', 'vt_up')


def coordinate(master):
    # Keep BUF/BUF-f, INV/CKINVDC and other structural prefixes distinct.
    match = re.fullmatch(r'(.+?)x(\d+(?:p\d+)?|p\d+)_ASAP7_75t_(SL|L|R)', master)
    if not match:
        return None
    return match[1], float(match[2].replace('p', '.')), VT[match[3]]


def forbidden(master):
    return any(fnmatch.fnmatchcase(master, pattern) for pattern in
               ('*x1p*_ASAP7*', '*xp*_ASAP7*', 'SDF*', 'ICG*'))


@dataclass
class ActionCatalog:
    """Fixed technology: official groups and pin semantics, loaded once per worker."""
    library: dict
    groups: dict

    @classmethod
    def load(cls, library_path, group_file=None):
        library = legality.read_library(str(library_path))
        group_file = group_file or ROOT / 'evaluator/third_party/displacement/asap7_equivalent_cell_list.csv'
        groups = {}
        with Path(group_file).open() as stream:
            for number, group in enumerate(csv.reader(stream)):
                for cell in group:
                    if cell in groups:
                        raise ValueError('Duplicate official equivalence member: ' + cell)
                    groups[cell] = (number, group)
        return cls(library, groups)


def enumerate_actions(design, catalog):
    """Four decisions per instance; nearest legal drive/VT neighbor, no I/O.

    design.views requires LEF + BC/TC/WC. Functional/PG/footprint policy is
    checked again by the public evaluator.
    """
    library, groups, views = catalog.library, catalog.groups, design.views
    result = []
    for name, cell in sorted(design.cells.items()):
        current = cell['master']
        coord = coordinate(current)
        reason = ('protected' if name in design.protected else
                  'unsupported_coordinate' if coord is None else
                  'missing_official_group' if current not in groups else None)
        for direction in DIRECTIONS:
            record = dict(instance=name, current=current, direction=direction,
                          target=None, valid=False, reason=reason, rejected=[])
            if reason:
                result.append(record)
                continue
            axis = 1 if direction.startswith('drive') else 2
            sign = 1 if direction.endswith('up') else -1
            choices = []
            group_id, members = groups[current]
            for target in members:
                other = coordinate(target)
                if (other is None or other[0] != coord[0] or
                    other[3-axis] != coord[3-axis] or sign * (other[axis] - coord[axis]) <= 0):
                    continue
                why = None
                if forbidden(target):
                    why = 'forbidden_target'
                elif not {'LEF', 'BC', 'TC', 'WC'} <= set(views.get(target, [])):
                    why = 'missing_view'
                elif target not in library or set(library[target]) != set(library.get(current, {})):
                    why = 'pin_contract'
                else:
                    for pin, original in library[current].items():
                        candidate = library[target][pin]
                        if any(original[k] != candidate[k] for k in ('direction', 'function', 'tristate')):
                            why = 'functional_contract'
                            break
                if why:
                    record['rejected'].append(dict(target=target, reason=why))
                else:
                    choices.append((abs(other[axis]-coord[axis]), target))
            if choices:
                target = min(choices)[1]
                record.update(target=target, valid=True, reason=None,
                              official_group=group_id, views=views[target])
            else:
                record['reason'] = 'no_legal_neighbor'
            result.append(record)
    return result
