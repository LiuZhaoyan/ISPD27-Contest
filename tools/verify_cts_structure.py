#!/usr/bin/env python3
"""Certify CTS-only logical equivalence using original gate-pin connectivity.

Original signal-bearing cells, port directions and every original pin must be
preserved. New cells may only be Liberty-classified identity buffers, or dummy
inverters whose outputs have no loads. After collapsing new identity buffers,
the partitions of all original pins into nets must be identical in both designs.
This checks both splits and shorts, independently of internal net names.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluator/internal'))
from legality import read_library, io, repeater
sys.path.insert(0,str(ROOT/'quad'))
from sta_server import tcl


def rows(path):
    with path.open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            # OpenDB escapes Verilog hierarchy separators on logical import,
            # while DEF import uses the literal instance/port names.
            for key in ('instance','pin'):
                if key in row:row[key]=re.sub(r'\\([/\\\[\]])',r'\1',row[key])
            yield row


def cells(prefix):
    result={}
    for r in rows(Path(str(prefix)+'.cells.tsv')):
        if r['physical']!='0':continue
        name=r['instance']
        if name in result:raise ValueError('Ambiguous normalized instance name: '+name)
        result[name]=(r['master'],r['physical'],r['macro'],r['x'],r['y'],r['orientation'],r['status'])
    return result


def verify(source,post,library,check_macro_locations=True):
    original=cells(source);current=cells(post)
    for name,cell in original.items():
        if name not in current or cell[0]!=current[name][0]:raise ValueError('Original cell removed or changed: '+name)
        if check_macro_locations and cell[2]=='1' and cell[3:]!=current[name][3:]:raise ValueError('Original macro moved: '+name)
    added=current.keys()-original.keys()
    additions=defaultdict(dict);net_degree=Counter()
    for r in rows(Path(str(post)+'.pins.tsv')):
        if r['signal'] in ('POWER','GROUND'):continue
        if r['net']:net_degree[r['net']]+=1
        if r['instance'] in added:additions[r['instance']][r['pin']]=r['net']
    parents={}
    def find(n):
        root=n
        while parents.get(root,root)!=root:root=parents[root]
        while n in parents and parents[n]!=n:
            nxt=parents[n];parents[n]=root;n=nxt
        return root
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[b]=a
    buffers=dummies=0
    for name in added:
        master=current[name][0];kind=repeater(library,master)
        inputs,outputs=io(library,master)
        if kind is None or len(inputs)!=1 or len(outputs)!=1:raise ValueError('Unexpected new logic: '+name)
        p=additions[name];a=p.get(inputs[0],'');y=p.get(outputs[0],'')
        if y and net_degree[y]>1:
            if kind!=0 or not a:raise ValueError('Non-identity insertion drives logic: '+name)
            union(a,y);buffers+=1
        else:dummies+=1
    source_pins={}
    for r in rows(Path(str(source)+'.pins.tsv')):
        if r['signal'] in ('POWER','GROUND') or (r['instance'] and r['instance'] not in original):continue
        key=(r['instance'],r['pin'])
        if key in source_pins:raise ValueError('Duplicate original pin')
        source_pins[key]=(r['net'],r['direction'])
    total=len(source_pins);mapping={};reverse={}
    for r in rows(Path(str(post)+'.pins.tsv')):
        if r['signal'] in ('POWER','GROUND') or (r['instance'] and r['instance'] not in original):continue
        key=(r['instance'],r['pin'])
        if key not in source_pins:raise ValueError('Unexpected original pin: '+str(key))
        old,direction=source_pins.pop(key);new=r['net']
        if direction!=r['direction'] or bool(old)!=bool(new):raise ValueError('Pin direction/connection changed: '+str(key))
        if not old:continue
        new=find(new)
        if old in mapping and mapping[old]!=new:raise ValueError('Original net split at '+str(key))
        if new in reverse and reverse[new]!=old:raise ValueError('Original nets shorted at '+str(key))
        mapping[old]=new;reverse[new]=old
    if source_pins:raise ValueError('Missing original pins')
    return dict(status='PASS',method='Exact original gate-pin net partitions modulo newly inserted Liberty identity buffers and output-unloaded dummy loads',
                original_signal_cells=len(original),original_signal_pins=total,original_connected_nets=len(mapping),
                added_identity_buffers=buffers,added_unloaded_dummies=dummies,original_macro_locations_preserved=True if check_macro_locations else None,
                sat_proof=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--benchmark-dir',type=Path)
    p.add_argument('--source-authority',choices=('paired','def'),default='paired',help='def: certify the physical source only; raw source Verilog is retained but not claimed equivalent')
    p.add_argument('--openroad',type=Path,default=ROOT/'tools/contest-runtime/install/OpenROAD/bin/openroad')
    args=p.parse_args();bench=(args.benchmark_dir or ROOT/'benchmarks'/args.benchmark).resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    platform=ROOT/'platform/asap7'
    common=f'set ::env(PLATFORM_DIR) {tcl(platform)}\nsource {tcl(ROOT/"evaluator/internal/legality_snapshot.tcl")}\n'
    common+=f'source {tcl(platform/"util/liberty_suppressions.tcl")}\nsource {tcl(platform/"util/libraries.tcl")}\n'
    for lef in ('asap7_tech_1x_201209.lef','asap7sc7p5t_28_R_1x_220121a.lef','asap7sc7p5t_28_L_1x_220121a.lef','asap7sc7p5t_28_SL_1x_220121a.lef'):
        common+=f'read_lef {tcl(platform/"lef"/lef)}\n'
    common+=f'source {tcl(bench/"platform_extra.tcl")}\nforeach lef $benchmark_extra_lefs {{read_lef $lef}}\nforeach lib $corner_libs(TC) {{read_liberty $lib}}\n'
    started=time.monotonic()
    for label,path in (('source',bench/'source/contest.def'),('post',bench/'input.def')):
        script=out/f'{label}.tcl';script.write_text(common+f'read_def {tcl(path)}\nlegality_snapshot {tcl(out/label)} 1\n')
        with (out/f'{label}.log').open('w') as stream:
            subprocess.run([str(args.openroad),'-no_init','-no_splash','-exit',str(script)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    library=read_library(out/'post.library.tsv')
    result=verify(out/'source',out/'post',library)
    if args.source_authority=='paired':
        top=json.loads((bench/'benchmark.json').read_text())['top_module']
        script=out/'source_verilog.tcl'
        script.write_text(common+f'read_verilog {tcl(bench/"source/functional.v")}\nlink_design {tcl(top)}\nlegality_snapshot {tcl(out/"source_verilog")} 1\n')
        with (out/'source_verilog.log').open('w') as stream:
            subprocess.run([str(args.openroad),'-no_init','-no_splash','-exit',str(script)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        source_consistency=verify(out/'source_verilog',out/'source',library,check_macro_locations=False)
        result['source_def_verilog_consistency']=source_consistency
    else:
        result['source_def_verilog_consistency']={'status':'NOT_CLAIMED','source_authority':'DEF','reason':'Original BSG Verilog aliases an internal net to a distinct clock input; physical DEF is authoritative'}
    result['benchmark']=args.benchmark;result['seconds']=time.monotonic()-started
    result['input_sha256']={s:hashlib.sha256((bench/s).read_bytes()).hexdigest() for s in ('source/contest.def','input.def','input.v')}
    (out/'equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(args.benchmark,'CTS STRUCTURAL EQUIVALENCE PASS',flush=True)

if __name__=='__main__':main()
