"""Independent GR reference and transactional physical edits for post-CTS flow."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
from forward_sta import rows
from native_metrics import checkpoint_code, read_checkpoint
from native_repair import repair_code
from sta_server import ROOT, CORNERS, Server, load_design, ROUTE, ROUTING_CONFIG, swaps, tcl, export_instances, output_design


class NativeWorkflow:
    def __init__(self, benchmark, out, threads=8):
        self.benchmark = benchmark; self.out = Path(out); self.threads = threads
        self.out.mkdir(parents=True, exist_ok=True)
        self.server = None; self.serial = 0; self.original_xy = self.out/'original_xy.tcl'
        if 'OPENROAD_EXE' not in os.environ:
            os.environ['OPENROAD_EXE'] = str(ROOT/'tools/contest-runtime/install/OpenROAD/bin/openroad')
        self.load()
        self.eligible=[r['odb_name'] for r in rows(self.out/'instances.tsv')
                       if r['status']=='PLACED' and r['frozen']!='1']

    def load(self, design=None):
        if self.server: self.server.close()
        self.serial += 1
        self.server = Server(self.out/f'session_{self.serial:03}', threads=self.threads)
        code = load_design(self.benchmark,self.out,design_def=design,route=False)
        # Placement pin-access R depends on the allowed routing layers too.
        # Apply the same settings before every initial/restored estimate, not
        # only on the first GR call (which otherwise changes every net's RC).
        code += ROUTING_CONFIG
        code += f'source {tcl(ROOT/"quad/export_placement_rc.tcl")}\n'
        if self.original_xy.exists(): code += f'source {tcl(self.original_xy)}\n'
        else:
            code += '''set quad_original_xy [dict create]
foreach i [[ord::get_db_block] getInsts] {dict set quad_original_xy [$i getName] [$i getLocation]}
'''+f'set f [open {tcl(self.original_xy)} w]\nputs $f [list set quad_original_xy $quad_original_xy]\nclose $f\n'
        self.server.call(code)

    def snapshot(self, path):
        path = Path(path); path.mkdir(parents=True,exist_ok=True)
        self.server.call(f'write_def {tcl(path/"model.def")}\n'+export_instances(path/'instances.tsv')+
                         f'placement_rc_export {tcl(path/"pins_geometry.tsv")}\n')
        return path

    def restore(self, snapshot):
        self.load(Path(snapshot)/'model.def')

    def repair(self, *, erc=True, hold=False, iterations=20, slew_margin=5., routed=False, corners=None):
        # Setup optimization belongs to the GPU controller. Native buffering is
        # reserved for electrical legality / hold closure, and timed separately.
        code = ''
        for c in (corners or ('WC','TC','BC')):
            part=repair_code(setup=False,hold=hold,erc=erc,iterations=iterations,corner=c,slew_margin=slew_margin)
            if routed:part=part.replace('estimate_parasitics -placement','estimate_parasitics -global_routing')
            code+=part
        self.server.call(code)

    def apply(self, actions):
        sizing = [a for a in actions if a['kind']=='size']
        code = swaps(sizing,update_parasitics=False)
        for a in actions:
            if a['kind']!='buffer': continue
            loads = ' '.join(tcl(p) for p in a['sinks'])
            x,y = a['location']
            code += (f'insert_buffer -buffer_cell {tcl(a["target"])} -load_pins [get_pins -quiet [list {loads}]] '
                     f'-location {{{x:.9f} {y:.9f}}} -buffer_name {tcl(a["instance"])} '
                     f'-net_name {tcl(a["instance"]+"_net")}\n')
        code += '''detailed_placement
check_placement -verbose
if {[clock_snapshot] ne $original_clock} {error "Workflow changed protected clock tree or sinks"}
'''
        return self.server.call(code)

    def reference(self, path, models=True, timing_only=False):
        """Fresh independent routing; native placement SPEF is a separate product.

        The reference STA never reads a custom estimated RC network. Native
        placement SPEFs are generated before GR, and only the GPU model uses
        their tree geometry/capacitance. No GR SPEF enters model construction.
        """
        started=time.perf_counter();path=Path(path);path.mkdir(parents=True,exist_ok=True)
        (path/'placement').mkdir(exist_ok=True);(path/'gr').mkdir(exist_ok=True)
        code = f'write_def {tcl(path/"model.def")}\n'+export_instances(path/'instances.tsv')
        code += f'placement_rc_export {tcl(path/"pins_geometry.tsv")}\n'
        if models:
            code += f'estimate_parasitics -placement -spef_file {tcl(path/"placement/network.spef")}\n'
        code += ROUTE+f'estimate_parasitics -global_routing -spef_file {tcl(path/"gr/network.spef")}\n'
        code += checkpoint_code(path/'resident_metrics',self.eligible)
        self.server.call(code)
        route_seconds=time.perf_counter()-started
        # A fresh reader of this exact GR SPEF is the timing authority, matching
        # the evaluator's final MCMM refresh. In-place master edits and native
        # estimated parasitics can otherwise disagree with the saved reference
        # on some JPEG setup endpoints even after delays/arrivals invalidation.
        oracle_started=time.perf_counter();oracle=Server(path/'oracle',threads=self.threads)
        try:
            oracle_code=load_design(self.benchmark,path/'oracle',design_def=path/'model.def',route=False,
                scene_spefs={c:path/f'gr/network_{c}.spef' for c in CORNERS})
            oracle_code+=checkpoint_code(path/'metrics',self.eligible,physical=False)
            if timing_only or models:
                oracle_code+=f'source {tcl(ROOT/"quad/workflow_export.tcl")}\n'
                for c in CORNERS:
                    for analysis in ('max','min'):
                        oracle_code+=f'workflow_endpoint_export {tcl(path/c/"audit"/analysis)} {c} {analysis}\n'
            oracle.call(oracle_code)
        finally:oracle.close()
        # Route usage and displacement belong to the parent physical design;
        # reloading DEF/SPEF does not reconstruct the router's usage arrays.
        shutil.copyfile(path/'resident_metrics/physical.tsv',path/'metrics/physical.tsv')
        metrics=read_checkpoint(path/'metrics')
        timing={'route_and_resident_metrics':route_seconds,'fresh_mcmm_wall':time.perf_counter()-oracle_started,
                'route_and_metrics':time.perf_counter()-started}
        if models:timing.update(self.complete_reference(path))
        timing['total']=time.perf_counter()-started
        (path/'reference.json').write_text(json.dumps(dict(metrics=metrics,seconds=timing,
            timing_reference='fresh_mcmm_from_independent_global_routing_spef',gpu_rc='openroad_native_placement'),indent=2)+'\n')
        return metrics,timing

    def complete_reference(self,path):
        """Export calibration data from saved GR, without running GR again.

        The parent must hold this snapshot (freshly checked or restored). The
        independent workers read saved GR SPEF; native placement RC stays a
        separate model input. A passing endpoint audit needs none of this work.
        """
        path=Path(path);started=time.perf_counter()
        # The marker is written only after every worker and agreement check
        # succeeds. Partial exports from a failed check must never be reused.
        if ((path/'endpoint_export_check.json').exists()
            and all((path/c/'placement/max/pins.tsv').exists() for c in CORNERS)):
            return dict(export_wall=0.,reused_export=True)
        (path/'placement').mkdir(exist_ok=True)
        if not all((path/f'placement/network_{c}.spef').exists() for c in CORNERS):
            self.server.call(f'estimate_parasitics -placement -spef_file {tcl(path/"placement/network.spef")}\n')
        def export(c):
            t=time.perf_counter();folder=path/c;folder.mkdir(exist_ok=True)
            child=Server(folder/'native',threads=max(1,self.threads//3))
            try:
                child.call(load_design(self.benchmark,folder,scenes=(c,),design_def=path/'model.def',
                                       spef=path/f'gr/network_{c}.spef')+
                    f'source {tcl(ROOT/"tools/export_analysis_graph.tcl")}\n'+
                    f'source {tcl(ROOT/"tools/export_source_groups.tcl")}\n'+
                    f'source {tcl(ROOT/"quad/workflow_export.tcl")}\n'+
                    f'workflow_export {tcl(folder)} {c} {tcl(path/f"placement/network_{c}.spef")}\n')
            finally:child.close()
            return c,time.perf_counter()-t
        with ThreadPoolExecutor(max_workers=3) as pool:corners=dict(pool.map(export,CORNERS))
        agreement={}
        for c in CORNERS:
            for analysis in ('max','min'):
                audit=path/c/'audit'/analysis/'endpoints.tsv'
                if not audit.exists():continue
                def values(file):
                    result={}
                    for row in rows(file):
                        key=(row['pin'],row['rf']);result[key]=min(result.get(key,float('inf')),float(row['slack_ps']))
                    return result
                light=values(audit);full=values(path/c/'gr'/analysis/'endpoints.tsv')
                error=max((abs(light[k]-full[k]) for k in light.keys()&full.keys()),default=0.)
                agreement[c+':'+analysis]=dict(missing=len(full.keys()-light.keys()),extra=len(light.keys()-full.keys()),max_abs_ps=error)
                if light.keys()!=full.keys() or error>.001:
                    raise RuntimeError('Lightweight and full native endpoint exports disagree: '+c+' '+analysis)
        (path/'endpoint_export_check.json').write_text(json.dumps(agreement,indent=2)+'\n')
        return dict(corner_exports=corners,export_wall=time.perf_counter()-started,native_endpoint_agreement=agreement)

    def output(self, path):
        self.server.call(output_design(path,self.benchmark))
        from def_output import preserve_input_grid
        preserve_input_grid(ROOT/'benchmarks'/self.benchmark/'input.def',Path(path)/(self.benchmark+'.def'))

    def final_reference(self,path,validated=None,metrics=None):
        """Reuse a fresh validated oracle only for a byte-identical design."""
        started=time.perf_counter();path=self.snapshot(path)
        if validated is not None and metrics is not None:
            validated=Path(validated)
            metadata=json.loads((validated/'reference.json').read_text())
            same=all((path/name).read_bytes()==(validated/name).read_bytes()
                     for name in ('model.def','instances.tsv','pins_geometry.tsv'))
            if (same and metadata.get('metrics')==metrics
                and metadata.get('timing_reference')=='fresh_mcmm_from_independent_global_routing_spef'):
                for name in ('gr','metrics'):shutil.copytree(validated/name,path/name)
                timing=dict(total=time.perf_counter()-started,reused_validated_reference=str(validated))
                (path/'reference.json').write_text(json.dumps(dict(metadata,seconds=timing),indent=2)+'\n')
                return metrics,timing
        return self.reference(path,models=False)

    def placement(self,path,previous_geometry=None):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        seconds=self.server.call(f'placement_rc_export {tcl(path/"pins_geometry.tsv")}\n'+export_instances(path/'instances.tsv'))
        same=previous_geometry is not None and Path(previous_geometry).read_bytes()==(path/'pins_geometry.tsv').read_bytes()
        code='estimate_parasitics -placement'
        if not same:code+=f' -spef_file {tcl(path/"network.spef")}'
        seconds+=self.server.call(code+'\n')
        return dict(seconds=seconds,geometry_reused=same)

    def electrical(self,path):
        from native_metrics import parse_erc_violations
        path=Path(path);path.mkdir(parents=True,exist_ok=True);code=''
        kinds={'max_slew':'slew_violation_sum_ps','max_capacitance':'cap_violation_sum_ff','max_fanout':'fanout_violation_sum'}
        for c in CORNERS:
            for kind in kinds:
                code+=f'report_check_types -scenes {c} -{kind} -violators -digits 9 -no_line_splits > {tcl(path/(c+"_"+kind+".rpt"))}\n'
        self.server.call(code)
        return {c:{key:parse_erc_violations((path/(c+'_'+kind+'.rpt')).read_text(),kind)['sum_violation']
                   for kind,key in kinds.items()} for c in CORNERS}

    def close(self):
        if self.server: self.server.close();self.server=None
