"""Persistent native STA process; optimizer and evaluator remain separate."""
import os, queue, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTEST=ROOT
CORNERS=('BC','TC','WC')

def tcl(value):
    value=str(value)
    if any(c in value for c in '{}\n'):raise ValueError('Unsupported Tcl path/name')
    return '{'+value+'}'

class Server:
    def __init__(self,out,binary=None,threads=8):
        binary=binary or os.environ.get('OPENROAD_EXE','openroad')
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.serial=0;self.queue=queue.Queue();self.log=(self.out/'openroad.log').open('w')
        script=self.out/'server.tcl'
        script.write_text('''while {[gets stdin command] >= 0} {
 if {[catch {source $command} message options]} {
   puts "QUAD_ERROR $message"; puts [dict get $options -errorinfo]
 } else {puts QUAD_OK}
 flush stdout
}
exit
''')
        self.proc=subprocess.Popen([binary,'-no_init','-no_splash','-threads',str(threads),'-exit',str(script)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=ROOT)
        def read():
            for line in self.proc.stdout:self.log.write(line);self.log.flush();self.queue.put(line.rstrip())
            self.queue.put(None)
        self.reader=threading.Thread(target=read,daemon=True);self.reader.start()

    def call(self,code):
        self.serial+=1;path=self.out/f'command_{self.serial:04}.tcl';path.write_text(code)
        self.proc.stdin.write(str(path)+'\n');self.proc.stdin.flush();start=time.perf_counter()
        while True:
            try:line=self.queue.get(timeout=60)
            except queue.Empty:
                # Silence is not a failure; keep a bounded wall-time limit.
                if time.perf_counter()-start>=18000:
                    raise RuntimeError('STA exceeded five hours; '+str(self.out/'openroad.log'))
                continue
            if line is None:raise RuntimeError('STA exited; '+str(self.out/'openroad.log'))
            if line=='QUAD_OK':return time.perf_counter()-start
            if line.startswith('QUAD_ERROR'):raise RuntimeError(line+'; '+str(self.out/'openroad.log'))

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=10)
        self.reader.join(timeout=2);self.log.close()

ROUTING_CONFIG='''set_scene TC
puts QUAD_STAGE_ROUTE
set_routing_layers -signal M2-M7 -clock M4-M7
set_global_routing_layer_adjustment * 0.25
set_global_routing_random -seed 42
'''

ROUTE=ROUTING_CONFIG+'''
global_route -allow_congestion -congestion_iterations 30
estimate_parasitics -global_routing
'''

def export_instances(path):
    return f'set f [open {tcl(path)} w]\n'+'''puts $f "name\todb_name\tmaster\tstatus\tfrozen"
set sta_names [dict create]
foreach cell [get_cells -hierarchical *] {
 set dbinst [sta::sta_to_db_inst $cell]
 if {$dbinst ne "NULL"} {dict set sta_names [$dbinst getName] [get_full_name $cell]}
}
foreach inst [[ord::get_db_block] getInsts] {
 set dbname [$inst getName]
 set name $dbname
 if {[dict exists $sta_names $dbname]} {set name [dict get $sta_names $dbname]}
 puts $f "$name\t$dbname\t[[$inst getMaster] getName]\t[$inst getPlacementStatus]\t[$inst isDoNotTouch]"
}
close $f
'''

SAVE='''set saved_placement {}
foreach inst [[ord::get_db_block] getInsts] {
 lappend saved_placement [list [$inst getName] [$inst getMaster] [$inst getLocation] [$inst getPlacementStatus] [$inst getOrient]]
}
'''

RESTORE='''set block [ord::get_db_block]
foreach item $saved_placement {
 lassign $item name master loc status orient
 set inst [$block findInst $name]
 if {[$inst getMaster] != $master} {
  if {![$inst swapMaster $master]} {error "restore master failed"}
 }
 $inst setOrient $orient; $inst setLocation {*}$loc; $inst setPlacementStatus $status
}
'''+ROUTE

def load_design(benchmark,out,scenes=('TC','BC','WC'),design_def=None,spef=None,route=True,scene_spefs=None):
    if scene_spefs is not None and (spef is not None or set(scene_spefs)!=set(scenes)):
        raise ValueError('Explicit per-scene SPEFs must match the scenes')
    bench=CONTEST/'benchmarks'/benchmark;platform=CONTEST/'platform/asap7'
    # Reuse organizer clock protection and physical-cell handling verbatim.
    definitions=(CONTEST/'evaluator/internal/evaluation.tcl').read_text().split('source $::env(MCMM_CONFIG)')[0]
    definitions=definitions.replace('source [file join [file dirname [info script]] legality_snapshot.tcl]',
                                    'source '+tcl(CONTEST/'evaluator/internal/legality_snapshot.tcl'))
    code=f'set ::env(RUN_RESIZER) 0\nset ::env(PLATFORM_DIR) {tcl(platform)}\n'+definitions
    code+=f'source {tcl(platform/"util/liberty_suppressions.tcl")}\nsource {tcl(platform/"util/libraries.tcl")}\n'
    for name in ('asap7_tech_1x_201209.lef','asap7sc7p5t_28_R_1x_220121a.lef','asap7sc7p5t_28_L_1x_220121a.lef','asap7sc7p5t_28_SL_1x_220121a.lef'):
        code+=f'read_lef {tcl(platform/"lef"/name)}\n'
    extra_platform = bench/'platform_extra.tcl'
    if extra_platform.exists():
        code+=f'source {tcl(extra_platform)}\nforeach lef $benchmark_extra_lefs {{read_lef $lef}}\n'
    # The first loaded corner defines the linked timing graph. This order is
    # material when library conditional groups differ; match core_run.sh.
    code+='puts QUAD_STAGE_LIBRARIES\nset loaded_libs [dict create]\nforeach c {TC BC WC} {foreach lib $corner_libs($c) {if {![dict exists $loaded_libs $lib]} {read_liberty $lib; dict set loaded_libs $lib 1}}}\n'
    code+=f'puts QUAD_STAGE_DEF\nread_def {tcl(design_def or bench/"input.def")}\nsource {tcl(bench/"mcmm.tcl")}\n'
    if scene_spefs is not None:
        for c,path in scene_spefs.items():code+=f'set quad_scene_spef({c}) {tcl(path)}\n'
    code+='foreach c {'+' '.join(scenes)+'''} {
 set mode $scenario_mode($c)
 read_sdc -mode $mode $scenario_sdc($c)
 set_mode $mode
 foreach clk [all_clocks] {if {![get_property $clk is_virtual]} {set_propagated_clock $clk}}
'''
    if spef:
        if len(scenes)!=1:raise ValueError('Reference SPEF requires one scene')
        code+=f'read_spef -name quad_rc {tcl(spef)}\ndefine_scene $c -mode $mode -liberty $corner_libs($c) -spef quad_rc\n'
    elif scene_spefs is not None:
        code+='read_spef -name quad_rc_$c $quad_scene_spef($c)\ndefine_scene $c -mode $mode -liberty $corner_libs($c) -spef quad_rc_$c\n'
    else:code+='define_scene $c -mode $mode -liberty $corner_libs($c)\n'
    code+='}\nset_scene '+('TC' if 'TC' in scenes else scenes[0])+'\n'
    code+=f'source {tcl(platform/"util/setRC.tcl")}\n'
    code+='set_dont_use {*x1p*_ASAP7* *xp*_ASAP7* SDF* ICG*}\nputs QUAD_STAGE_CLOCK_PROTECTION\nset original_clock [clock_snapshot]\nprotect_clocks\ncheck_placement\nputs QUAD_STAGE_INSTANCES\n'
    code+=f'set f [open {tcl(Path(out)/"instances.tsv")} w]\n'
    code+='''puts $f "name\todb_name\tmaster\tstatus\tfrozen"
set sta_names [dict create]
foreach cell [get_cells -hierarchical *] {
 set dbinst [sta::sta_to_db_inst $cell]
 if {$dbinst ne "NULL"} {dict set sta_names [$dbinst getName] [get_full_name $cell]}
}
foreach inst [[ord::get_db_block] getInsts] {
 set dbname [$inst getName]
 set name $dbname
 if {[dict exists $sta_names $dbname]} {set name [dict get $sta_names $dbname]}
 puts $f "$name\t$dbname\t[[$inst getMaster] getName]\t[$inst getPlacementStatus]\t[$inst isDoNotTouch]"
}
close $f
'''
    if not spef and scene_spefs is None and route:code+=ROUTE
    code+=f'source {tcl(ROOT/"quad/export_paper.tcl")}\n'
    return code

def export_reference(server,path,benchmark):
    """Exact per-scene reference reloads sharing the same native routed snapshot.

    All libraries still load TC/BC/WC in canonical order. Only one constraint
    scene is instantiated in each exporter, so arrival iteration needs no tag
    formatting. Native candidate acceptance continues to use simultaneous MCMM.
    """
    path=Path(path).resolve();path.mkdir(parents=True,exist_ok=True)
    server.call(f'write_def {tcl(path/"model.def")}\nestimate_parasitics -global_routing -spef_file {tcl(path/"network.spef")}\n')
    export_reference_workers(path,benchmark)

def export_reference_workers(path,benchmark):
    """Export corners from a fresh routed DEF/SPEF snapshot of this run."""
    path=Path(path).resolve()
    def export_corner(c):
        # Three independent exporters use at most twelve native CPU threads;
        # the parent MCMM process is idle while they read the same snapshot.
        child=Server(path/c/'sta',threads=4)
        try:
            child.call(load_design(benchmark,path/c,scenes=(c,),design_def=path/'model.def',spef=path/('network_'+c+'.spef'))
                       +f'paper_export {tcl(path/c)} {tcl(path/("network_"+c+".spef"))}\n')
        finally:child.close()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(export_corner,CORNERS))

def export_models(path):
    path=Path(path)
    code=f'file mkdir {tcl(path)}\nestimate_parasitics -global_routing -spef_file {tcl(path/"network.spef")}\n'
    return code+''.join(f'set_scene {c}\npaper_export {tcl(path/c)} {tcl(path/("network_"+c+".spef"))}\n' for c in CORNERS)

def metrics(path):
    code=f'set f [open {tcl(path)} w]\n'
    code+='''puts $f "corner\tsetup_tns_ps\tsetup_wns_ps\thold_tns_ps\thold_wns_ps\tinternal_w\tswitching_w\tleakage_w\ttotal_w"
foreach c {BC TC WC} {
 set_scene $c
 sta::find_requireds
 sta::unset_power_input_activity
 set values [list $c]
 foreach delay {max min} {
  lappend values [expr {[sta::total_negative_slack_scene_cmd [sta::cmd_scene] $delay]*1e12}]
  lappend values [expr {[sta::worst_slack_scene [sta::cmd_scene] $delay]*1e12}]
 }
 lappend values {*}[lrange [sta::design_power [sta::cmd_scene]] 0 3]
 puts $f [join $values "\t"]
}
close $f
'''
    return code

def reload_named_parasitics(path):
    """Refresh the named scene networks after a routed geometry change."""
    path=Path(path).resolve();path.mkdir(parents=True,exist_ok=True)
    code=f'estimate_parasitics -global_routing -spef_file {tcl(path/"network.spef")}\n'
    for c in CORNERS:
        code+=f'read_spef -name quad_rc_{c} {tcl(path/("network_"+c+".spef"))}\n'
    return code

def swaps(actions,reverse=False,update_parasitics=True):
    code='set db [ord::get_db]\nset block [ord::get_db_block]\n'
    for a in actions:
        target=a['old'] if reverse else a['target']
        dbname=tcl(a.get('odb_name',a['instance']))
        code+=f'if {{![[$block findInst {dbname}] swapMaster [$db findMaster {tcl(target)}]]}} {{error "swap failed"}}\n'
    return code+('estimate_parasitics -global_routing\n' if update_parasitics else '')

def output_design(path,benchmark):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    return f'''if {{[clock_snapshot] ne $original_clock}} {{error "Clock tree or sinks changed"}}
check_placement
write_def {tcl(path/(benchmark+'.def'))}
write_verilog -remove_cells [physical_only] {tcl(path/(benchmark+'.v'))}
'''
