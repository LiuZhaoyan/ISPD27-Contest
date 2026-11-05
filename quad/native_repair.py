"""Explicit organizer Resizer sequence for contest constraint repair.

The sequence is copied from the released evaluator, not inferred from defaults.
Clock registers keep their master and placement while data pins allow repeaters.
"""


def repair_code(setup=True, hold=True, erc=True, iterations=500, corner='TC', slew_margin=None):
    if corner not in ('TC','BC','WC'):raise ValueError('Unknown native repair corner')
    if slew_margin is not None and not 0<=float(slew_margin)<100:raise ValueError('Invalid slew margin')
    code = f'''set_scene {corner}
set ::env(RUN_RESIZER) 1
protect_clocks
estimate_parasitics -placement
'''
    if erc:
        code += ('repair_design\n' if slew_margin is None else f'repair_design -slew_margin {float(slew_margin)}\n')
    if setup:
        code += f'''repair_timing -setup -sequence {{vt_swap sizeup swap buffer}} \
 -skip_gate_cloning -skip_buffer_removal -skip_last_gasp -repair_tns 100 \
 -max_iterations {int(iterations)} -verbose
'''
    if hold:
        code += f'repair_timing -hold -max_iterations {int(iterations)} -verbose\n'
    return code + '''detailed_placement
check_placement -verbose
if {[clock_snapshot] ne $original_clock} {error "Repair changed clocks or sink placement"}
set ::env(RUN_RESIZER) 0
protect_clocks
'''
