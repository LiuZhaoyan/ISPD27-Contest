# Single-clock max analysis with explicit input / launch-edge selectors.
# This exports fresh native evidence, not fitted candidate parameters.
proc source_group_export {directory scene {analysis max}} {
    if {$analysis ni {min max}} {error "Expected min/max source groups"}
    file mkdir $directory
    set_scene $scene
    sta::find_requireds
    set clocks {}
    foreach candidate [all_clocks] {
        if {![get_property $candidate is_virtual]} {lappend clocks $candidate}
    }
    if {[llength $clocks]!=1} {error "Source-group experiment requires one physical clock"}
    set clock [lindex $clocks 0]
    if {[get_property $clock is_generated]} {error "Generated clock requires explicit tags"}
    set f [open $directory/endpoints.tsv w]
    puts $f "source_group\tpin\trf\trole\tis_check\tis_path_delay\tat_ps\traw_at_ps\trequired_ps\tslack_ps\tmargin_ps\tcapture_pin\tcapture_rf\tcapture_at_ps\tstartpoint"
    foreach group {input clock_rise clock_fall} {
        if {$group eq "input"} {
            set selector [list -from [all_inputs -no_clocks]]
        } elseif {$group eq "clock_rise"} {
            set selector [list -rise_from $clocks]
        } else {set selector [list -fall_from $clocks]}
        foreach rf {rise fall} {
            foreach end [find_timing_paths -scenes $scene -path_delay $analysis {*}$selector -${rf}_to [sta::endpoints] -group_path_count 100000 -endpoint_path_count 1] {
                set transition [$end end_transition]
                if {$transition eq "^"} {set transition rise}
                if {$transition eq "v"} {set transition fall}
                set role [$end check_role]
                if {![$end is_check] && ([$end is_output_delay] || [$end is_path_delay])} {
                    set role [expr {$analysis eq "max" ? "output setup" : "output hold"}]
                }
                set capture_pin "";set capture_rf "";set capture_at 0
                set clock_path [$end target_clk_path]
                if {$clock_path ne "NULL"} {
                    set capture_pin [get_full_name [$clock_path pin]];set capture_rf [$clock_path edge]
                    if {$capture_rf eq "^"} {set capture_rf rise}
                    if {$capture_rf eq "v"} {set capture_rf fall}
                    set capture_at [expr {[$clock_path arrival]*1e12}]
                }
                set path [$end path]
                puts $f [join [list $group [get_full_name [$end pin]] $transition $role [$end is_check] [$end is_path_delay] \
                    [expr {[$end data_arrival_time]*1e12}] [expr {[$path arrival]*1e12}] \
                    [expr {[$end data_required_time]*1e12}] [expr {[$end slack]*1e12}] [expr {[$end margin]*1e12}] \
                    $capture_pin $capture_rf $capture_at [get_full_name [get_property $end startpoint]]] "\t"]
            }
        }
        puts "SOURCE_GROUP_EXPORTED $scene $group";flush stdout
    }
    close $f
}
