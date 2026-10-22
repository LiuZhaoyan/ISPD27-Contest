"""Physical tools, fresh native feedback, and required structural checks.

No objective, best-state bookkeeping, or candidate selection lives here.
"""
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time
from types import MappingProxyType, ModuleType

from data_io import ROOT, rows, save
sys.path.insert(0, str(ROOT/'evaluator'))
from internal import legality
from internal.metrics import parse_erc_violations


@dataclass(frozen=True)
class PhysicalResult:
    path: Path
    summary: dict
    graph: Path


def native(folder):
    result={}
    for row in rows(Path(folder)/'native.tsv'):
        values=(float(row['tns_ps'])/1000,float(row['wns_ps'])/1000)
        if not all(math.isfinite(v) for v in values): raise ValueError('Nonfinite native timing')
        key=(row['scene'],row['analysis'])
        if key in result: raise ValueError('Duplicate native MCMM statistics')
        result[key]=values
    if set(result)!={(s,a) for s in ('BC','TC','WC') for a in ('max','min')}:
        raise ValueError('Missing native MCMM statistics')
    return result


def native_baseline(folder):
    """Initial loss/guard reference, not a certified evaluator summary."""
    timing=native(folder)
    baseline={}
    for scene in ('BC','TC','WC'):
        row={}
        for analysis,check in (('max','setup'),('min','hold')):
            tns,wns=timing[(scene,analysis)]
            if tns>0: raise ValueError('Positive native TNS')
            row.update({check+'_tns_ns':tns,check+'_wns_ns':min(0.,wns)})
        for kind,metric in (('max_slew','slew_violation_sum_ps'),
                            ('max_capacitance','cap_violation_sum_ff'),
                            ('max_fanout','fanout_violation_sum')):
            row[metric]=parse_erc_violations((Path(folder)/f'{scene}_{kind}.rpt').read_text(),kind)['sum_violation']
        baseline[scene]=row
    return baseline


def canonical_netlist(path):
    """Compare only this backend's flat OpenROAD exports, not arbitrary RTL."""
    text=re.sub(r'//[^\n]*|/\*.*?\*/', '', Path(path).read_text(), flags=re.S)
    return sorted(tuple(re.findall(r"\\\S+|[A-Za-z_$][\w$]*|\d+'[bsdhBSHD][\da-fA-FxXzZ_]+|\d+|[^\s]", s))
                  for s in text.split(';') if s.strip())


# Compile trusted repository rules once, without writing evaluator bytecode.
# Each check executes them in its own namespace; public checker globals stay intact.
_RULES_PATH=Path(legality.__file__)
_RULES_CODE=compile(_RULES_PATH.read_text(),str(_RULES_PATH),'exec')


@dataclass(frozen=True)
class ParsedSnapshot:
    prefix: str
    records: object
    cells: tuple
    pins: tuple

    @classmethod
    def load(cls,prefix):
        prefix=str(prefix)
        records=MappingProxyType({key:tuple(value)
            for key,value in legality.def_records(prefix+'.def').items()})
        return cls(prefix,records,
            tuple(MappingProxyType(row) for row in legality.table(prefix+'.cells.tsv')),
            tuple(MappingProxyType(row) for row in legality.table(prefix+'.pins.tsv')))


@dataclass(frozen=True)
class ValidationContext:
    """One transaction's immutable data; never a cache of paths across calls."""
    before: ParsedSnapshot
    after: ParsedSnapshot
    library: object
    groups: object
    library_path: str
    group_file: str

    @classmethod
    def load(cls,before,after,library_path,group_file):
        old=ParsedSnapshot.load(before)
        new=old if str(before)==str(after) else ParsedSnapshot.load(after)
        library=MappingProxyType({master:MappingProxyType({pin:MappingProxyType(row)
            for pin,row in pins.items()})
            for master,pins in legality.read_library(library_path).items()})
        groups={}
        for index,line in enumerate(Path(group_file).read_text().splitlines()):
            for master in line.split(','):
                if master.strip(): groups[master.strip()]=index
        return cls(old,new,library,MappingProxyType(groups),str(library_path),str(group_file))

    def advance(self,after):
        """Reuse the middle snapshot and explicit catalog for the next boundary."""
        return ValidationContext(self.after,ParsedSnapshot.load(after),self.library,
                                 self.groups,self.library_path,self.group_file)

    def check(self,post_route=False):
        # Reuse unchanged public rule code, not a fork of its assertions. Only
        # file readers in this private module see the parsed snapshot records.
        snapshots={s.prefix:s for s in (self.before,self.after)}
        tables={prefix+suffix:getattr(snapshot,attr)
            for prefix,snapshot in snapshots.items()
            for suffix,attr in (('.cells.tsv','cells'),('.pins.tsv','pins'))}
        records={prefix+'.def':snapshot.records for prefix,snapshot in snapshots.items()}

        def table(path):
            return tables[str(path)]

        def def_records(path):
            # immutable_def pops/rewrites sections: never hand it the cache map.
            return {key:list(value) for key,value in records[str(path)].items()}

        def read_library(path):
            if str(path)!=self.library_path:
                raise ValueError('Unexpected validation library: '+str(path))
            return self.library

        rules=ModuleType('_quad_gradient_legality')
        rules.__file__=str(_RULES_PATH)
        try:
            exec(_RULES_CODE,rules.__dict__)
            rules.table,rules.def_records,rules.read_library=table,def_records,read_library
            return rules.check(self.before.prefix,self.after.prefix,
                               self.library_path,self.group_file,post_route)
        finally:
            # Break private function/global cycles promptly for large snapshots.
            rules.__dict__.clear()

    def validate(self,post_route=False):
        """Reuse public structural rules; additionally restrict this tool to sizing."""
        report=self.check(post_route)
        old={r['instance']:r for r in self.before.cells}
        new={r['instance']:r for r in self.after.cells}
        if old.keys()!=new.keys(): raise ValueError('Sizing changed instance set')
        # Counter preserves duplicate-row multiplicity without sorting all pins.
        if Counter(tuple(sorted(r.items())) for r in self.before.pins) != Counter(
                tuple(sorted(r.items())) for r in self.after.pins):
            raise ValueError('Sizing changed connectivity')
        changed=[name for name in old if old[name]['master']!=new[name]['master']]
        if changed:
            groups=self.groups
            library=self.library
            for name in changed:
                a,b=old[name]['master'],new[name]['master']
                if a not in groups or groups[a]!=groups.get(b):
                    raise ValueError('Sizing changed equivalence group: '+name)
                if a not in library or b not in library or library[a].keys()!=library[b].keys():
                    raise ValueError('Sizing changed library pins: '+name)
                for pin in library[a]:
                    for key in ('direction','function','tristate'):
                        if library[a][pin][key]!=library[b][pin][key]:
                            raise ValueError('Sizing changed pin semantics: '+name+'/'+pin+':'+key)
        return report


def validate_state(before, after, post_route=False):
    library_path=str(before)+'.library.tsv'
    group_file=ROOT/'evaluator/third_party/displacement/asap7_equivalent_cell_list.csv'
    return ValidationContext.load(before,after,library_path,group_file).validate(post_route)


def validate_annotation(folder):
    """Only snapshot-proven dangling drivers may lack clean STA parasitics."""
    folder=Path(folder)
    pins=list(rows(folder/'snapshot.pins.tsv'))
    drivers={}
    receivers=set()
    for pin in pins:
        inst,direction,net=pin['instance'],pin['direction'],pin['net']
        if direction in (('INPUT','INOUT') if inst else ('OUTPUT','INOUT')) and net:
            receivers.add(net)
        if direction in (('OUTPUT','INOUT') if inst else ('INPUT','INOUT')):
            name=inst+'/'+pin['pin'] if inst else pin['pin']
            if name in drivers: raise ValueError('Duplicate snapshot driver: '+name)
            drivers[name]=net
    coverage={}
    for scene in ('BC','TC','WC'):
        path=folder/f'{scene}_parasitic_annotation.rpt'
        try:
            lines=path.read_text().splitlines()
        except FileNotFoundError as exc:
            raise ValueError('Missing annotation report: '+scene) from exc
        if not lines or not re.fullmatch(r'Found (0|[1-9][0-9]*) unannotated drivers\.',lines[0]):
            raise ValueError('Malformed annotation report: '+scene)
        count=int(lines[0].split()[1])
        if len(lines)!=count+2 or lines[-1]!='Found 0 partially unannotated drivers.':
            raise ValueError('Malformed or partially unannotated report: '+scene)
        names=[]
        for line in lines[1:-1]:
            if not re.fullmatch(r' \S+',line):
                raise ValueError('Malformed annotation driver: '+scene)
            name=line[1:]
            if name in names: raise ValueError('Duplicate annotation driver: '+name)
            if name not in drivers: raise ValueError('Unknown annotation driver: '+name)
            if drivers[name] and drivers[name] in receivers:
                raise ValueError('Unannotated driver has receivers: '+name)
            names.append(name)
        coverage[scene]=dict(unannotated_dangling_drivers=count,partially_unannotated_drivers=0)
    return coverage


def kill_tree(pid):
    """Freeze before walking Linux children, so timeout cleanup cannot race forks.

    Children retain the launcher's process group: an outer SIGKILL also reaches
    every tool. Do not create sessions or process groups here.
    """
    try:
        os.kill(pid, signal.SIGSTOP)
    except ProcessLookupError:
        return
    children = set()
    for task in Path(f'/proc/{pid}/task').glob('*'):
        try:
            children.update(int(p) for p in (task/'children').read_text().split())
        except FileNotFoundError:
            pass
    for child in children:
        kill_tree(child)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Runner:
    def __init__(self, workspace, benchmark):
        self.workspace=Path(workspace)
        self.benchmark=benchmark
        self.commands=[]
        self.deadline=None
        self.analysis_worker=None
        self.analysis_log=None
        self.analysis_closed=False

    def close(self):
        """Reap the persistent worker and any descendants, including on cancel."""
        self.analysis_closed=True
        process=self.analysis_worker
        try:
            if process is not None:
                kill_tree(process.pid)
                process.wait()
                process.stdin.close()
                process.stdout.close()
        finally:
            self.analysis_worker=None
            if self.analysis_log is not None:
                self.analysis_log.close()
                self.analysis_log=None

    def command(self, command, log, env=None, timeout=1800):
        start=time.monotonic()
        if self.deadline is not None:
            timeout=min(timeout,self.deadline-start)
            if timeout<=0: raise TimeoutError('Search time budget exhausted')
        record=dict(command=[str(x) for x in command],log=str(log),seconds=None,returncode=None)
        self.commands.append(record)
        print('Running:',log,flush=True)
        save(self.workspace/'commands.json',self.commands)
        try:
            with Path(log).open('x') as stream:
                with subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                      env=env,cwd=ROOT) as process:
                    try:
                        process.wait(timeout=timeout)
                    except BaseException:
                        # Stop evaluator children as well as their Python parent.
                        kill_tree(process.pid)
                        process.wait()
                        raise
                    returncode=process.returncode
            record['returncode']=returncode
        finally:
            record['seconds']=time.monotonic()-start
            save(self.workspace/'commands.json',self.commands)
        if returncode: raise RuntimeError(f'Command failed ({returncode}): {log}')

    def analyze(self, graph, platform_dir, reference):
        # One non-detached worker per run, with fixed technology. Only dynamic
        # graph/model/scores are rebuilt. The parent bounds startup and requests.
        if self.analysis_closed:
            raise RuntimeError('Analysis worker closed; start a new optimization run')
        start=time.monotonic()
        timeout=10800 if self.deadline is None else min(10800,self.deadline-start)
        command=[sys.executable, str(ROOT/'tools/quad_gradient/analyze.py'),
                 '--libdir', str(platform_dir/'lib/NLDM')]
        record=dict(command=command,log=str(graph/'analysis.log'),seconds=None,
                    returncode=None,graph=str(graph))
        self.commands.append(record)
        try:
            if timeout<=0: raise TimeoutError('Search time budget exhausted')
            save(self.workspace/'commands.json',self.commands)
            if self.analysis_worker is None:
                self.analysis_log=(self.workspace/'analysis-worker.log').open('x')
                self.analysis_worker=subprocess.Popen(command,stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,stderr=self.analysis_log,cwd=ROOT)
            process=self.analysis_worker
            process.stdin.write((json.dumps(dict(graph=str(graph),reference=list(reference)))+'\n').encode())
            process.stdin.flush()
            response=b''
            while b'\n' not in response:
                remaining=start+timeout-time.monotonic()
                if remaining<=0 or not select.select([process.stdout],[],[],remaining)[0]:
                    raise subprocess.TimeoutExpired(command,timeout)
                chunk=os.read(process.stdout.fileno(),4096)
                if not chunk: raise RuntimeError('Analysis worker exited; see analysis-worker.log')
                response+=chunk
            reply=json.loads(response)
            if reply != {'ok': True}:
                raise RuntimeError(reply.get('error','Invalid analysis worker response'))
            result=(json.loads((graph/'scores.json').read_text()),
                    json.loads((graph/'analysis.json').read_text()))
            record['returncode']=0
            return result
        except BaseException:
            self.close()
            raise
        finally:
            record['seconds']=time.monotonic()-start
            save(self.workspace/'commands.json',self.commands)

    def _backend(self, baseline, folder, **options):
        env=os.environ.copy()
        env.update(QG_ROOT=str(ROOT),QG_BASELINE=str(baseline),QG_OUTPUT=str(folder),
                   QG_INITIALIZE='0',QG_ROUTE='0',QG_ACTION='',QG_EXPORT_GRAPH='0',
                   MCMM_CONFIG=str(ROOT/'benchmarks'/self.benchmark/'mcmm.tcl'))
        env.update(options)
        self.command([os.environ.get('OPENROAD_EXE','openroad'),'-exit',str(ROOT/'tools/quad_gradient/backend.tcl')],
                     folder/'openroad.log',env)

    def probe(self, baseline, name, graph=False):
        folder=self.workspace/name
        folder.mkdir()
        self._backend(baseline,folder,QG_EXPORT_GRAPH=str(int(graph)))
        native(folder)
        return folder

    def initialize(self, input_dir, top):
        """Prepare one routed checkpoint; do not run the complete R0 evaluator."""
        folder=self.workspace/'initial'
        folder.mkdir()
        self._backend(input_dir,folder,QG_INITIALIZE='1',QG_TOP=top,
                      MCMM_CONFIG=str(input_dir/'mcmm.tcl'))
        for name in ['evaluated.def']+[f'parasitics_{s}.spef' for s in ('BC','TC','WC')]:
            if not (folder/name).is_file(): raise ValueError('Missing initial checkpoint file: '+name)
        self.command([sys.executable,str(ROOT/'evaluator/internal/check_spef.py')]+
                     [str(folder/f'parasitics_{s}.spef') for s in ('BC','TC','WC')],folder/'spef-check.log')
        save(folder/'state-check.json',validate_state(folder/'before',folder/'snapshot',True))
        return folder

    def evaluate(self, name, baseline, actions) -> PhysicalResult:
        """Apply and route once, then validate both boundaries before clean STA."""
        folder=self.workspace/name
        folder.mkdir()
        for action in actions:
            if any(c in action[key] for key in ('instance','target') for c in '\t\r\n'):
                raise ValueError('Invalid TSV action field')
        action_file=folder/'actions.tsv'
        action_file.write_text(''.join(a['instance']+'\t'+a['target']+'\n' for a in actions))
        save(folder/'actions.json',actions)
        self._backend(baseline,folder,QG_ROUTE='1',QG_ACTION=str(action_file),
            QG_TOP=json.loads((ROOT/'benchmarks'/self.benchmark/'benchmark.json').read_text())['top_module'])
        context=ValidationContext.load(folder/'before',folder/'evaluated',str(folder/'before.library.tsv'),
            ROOT/'evaluator/third_party/displacement/asap7_equivalent_cell_list.csv')
        state=dict(apply=context.validate(),post_route=context.advance(folder/'snapshot').validate(True))
        save(folder/'state-check.json',state)
        self.command([sys.executable,str(ROOT/'evaluator/internal/check_spef.py')]+
                     [str(folder/f'parasitics_{s}.spef') for s in ('BC','TC','WC')],folder/'spef-check.log')
        spef_report=json.loads((folder/'spef-check.log').read_text())
        if len(spef_report)!=3 or any(not math.isfinite(r[k]) or r[k]<0
                for r in spef_report for k in ('declared_cap_ff','exported_cap_ff')):
            raise ValueError('Invalid SPEF capacitance')
        overflow=list(rows(folder/'overflow.tsv'))
        if len(overflow)!=1: raise ValueError('Missing routing overflow')
        total,peak=(int(overflow[0][k]) for k in ('total','max'))
        if total<0 or peak<0 or peak>total: raise ValueError('Invalid routing overflow')
        for filename in ['evaluated.def','evaluated.v']+[f'parasitics_{s}.spef' for s in ('BC','TC','WC')]:
            if not (folder/filename).is_file(): raise ValueError('Missing fresh checkpoint file: '+filename)
        # This independent process covers the remaining DEF serialization/reload
        # boundary and reads only new SPEFs. Its STA/graph, never in-memory action
        # timing, supplies both physical feedback and model inputs.
        clean=self.probe(folder,name+'-sta',graph=True)
        if canonical_netlist(clean/'loaded.v')!=canonical_netlist(folder/'evaluated.v'):
            raise ValueError('Clean STA netlist differs from routed checkpoint')
        coverage=validate_annotation(clean)
        summary=native_baseline(clean)
        for metrics in summary.values():
            metrics.update(global_routing_overflow=total,max_global_routing_overflow=peak)
        save(folder/'feedback.json',dict(provenance='tool_private_fresh_route_clean_native_sta',
             certified=False,scenarios=summary,state_checks=state,annotation_coverage=coverage,
             clean_sta=str(clean)))
        return PhysicalResult(folder,summary,clean)

