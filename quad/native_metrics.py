"""Native checkpoint metrics using the organizer's ERC parser and score terms."""
import csv,json,sys
from pathlib import Path
from sta_server import ROOT,CONTEST,CORNERS,tcl,metrics

sys.path.insert(0,str(CONTEST/'evaluator'))
from internal.metrics import parse_erc_violations
from score import calculate,read_summary

def checkpoint_code(path,eligible,*,physical=True):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    code=metrics(path/'timing_power.tsv')
    for c in CORNERS:
        for kind in ('max_slew','max_capacitance','max_fanout'):
            code+=f'report_check_types -scenes {c} -{kind} -violators -digits 9 -no_line_splits > {tcl(path/(c+"_"+kind+".rpt"))}\n'
    # A fresh DEF/SPEF timing oracle has no router grid or usage state.
    # Its caller must supply physical.tsv from the actual routed process.
    if not physical:return code
    code+='''if {![info exists quad_original_xy]} {
 set quad_original_xy [dict create]
 foreach inst [[ord::get_db_block] getInsts] {dict set quad_original_xy [$inst getName] [$inst getLocation]}
}
set quad_movement 0
set block [ord::get_db_block]
'''
    code+='foreach name {'+' '.join(tcl(name) for name in eligible)+'} {\n'
    code+='''lassign [dict get $quad_original_xy $name] ox oy
 lassign [[$block findInst $name] getLocation] x y
 set quad_movement [expr {$quad_movement+abs($x-$ox)+abs($y-$oy)}]
}
set grid [$block getGCellGrid]
set quad_overflow 0;set quad_peak 0
foreach layer [[ord::get_db_tech] getLayers] {
 if {[$layer getRoutingLevel]<=0} {continue}
 for {set x 0} {$x<[llength [$grid getGridX]]} {incr x} {
  for {set y 0} {$y<[llength [$grid getGridY]]} {incr y} {
   set over [expr {max(0,[$grid getUsage $layer $x $y]-[$grid getCapacity $layer $x $y])}]
   set quad_overflow [expr {$quad_overflow+$over}];set quad_peak [expr {max($quad_peak,$over)}]
  }
 }
}
'''
    code+=f'set f [open {tcl(path/"physical.tsv")} w]\n'
    code+='puts $f "displacement_um\toverflow\tmax_overflow"\n'
    code+=f'puts $f "[expr {{$quad_movement/double({len(eligible)})/[[ord::get_db_tech] getDbUnitsPerMicron]}}]\t$quad_overflow\t$quad_peak"\nclose $f\n'
    return code

def read_checkpoint(path):
    path=Path(path);physical=next(csv.DictReader((path/'physical.tsv').open(),delimiter='\t'));result={}
    for row in csv.DictReader((path/'timing_power.tsv').open(),delimiter='\t'):
        c=row['corner'];values={k:float(v) for k,v in row.items() if k!='corner'}
        scene={k.replace('_ps','_ns'):values[k]/1000 for k in ('setup_tns_ps','setup_wns_ps','hold_tns_ps','hold_wns_ps')}
        scene.update(dynamic_power_w=values['internal_w']+values['switching_w'],leakage_power_w=values['leakage_w'],average_displacement_um=float(physical['displacement_um']),global_routing_overflow=float(physical['overflow']),max_global_routing_overflow=float(physical['max_overflow']))
        for kind,key in [('max_slew','slew_violation_sum_ps'),('max_capacitance','cap_violation_sum_ff'),('max_fanout','fanout_violation_sum')]:
            scene[key]=parse_erc_violations((path/(c+'_'+kind+'.rpt')).read_text(),kind)['sum_violation']
        result[c]=scene
    return result

class NativeScore:
    @staticmethod
    def provided_baseline(benchmark):
        return read_summary(CONTEST/'evaluator/reference_results'/benchmark/'baseline.csv')[1]
    def __init__(self,benchmark,baseline):
        self.baseline=baseline;self.config=json.loads((CONTEST/'evaluator/config/scoring_config.json').read_text())
        folder=CONTEST/'evaluator/reference_results'/benchmark
        _,self.reference=read_summary(folder/'resizer.csv');self.runtime=json.loads((folder/'runtimes.json').read_text())
    def evaluate(self,candidate,tool_seconds=0.,flow_seconds=None):
        runtime={**self.runtime,'tool_s':tool_seconds,'flow_s':tool_seconds if flow_seconds is None else flow_seconds}
        return calculate(self.baseline,candidate,self.reference,self.config,runtime)
