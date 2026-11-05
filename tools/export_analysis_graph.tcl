# Full native graph for a selected min/max analysis; no design edits.
proc analysis_graph_arrival {vertex scene rf mm single_plain_scene} {
    # Formatting a Path's Tag is very expensive in the pinned build. With one
    # scene and no generated clocks, every path has the requested scene.
    if {!$single_plain_scene} {return [comparison_arrival $vertex $scene $rf $mm]}
    if {$mm eq "max"} {set value -Inf} else {set value Inf}
    set paths [$vertex path_iterator $rf $mm]
    while {[$paths has_next]} {
        set path [$paths next]
        set arrival [expr {[$path arrival]*1e12}]
        if {$mm eq "max"} {set value [expr {max($value,$arrival)}]} else {set value [expr {min($value,$arrival)}]}
    }
    $paths finish
    return $value
}

proc analysis_graph_export {directory scene mm} {
    if {$mm ni {min max}} {error "Expected min or max analysis"}
    file mkdir $directory
    set_scene $scene
    sta::find_requireds
    set ptr [sta::cmd_scene]
    set single_plain_scene [expr {[llength [sta::scenes]]==1}]
    foreach clock [all_clocks] {
        if {[get_property $clock is_generated]} {set single_plain_scene 0}
    }
    set pf [open $directory/pins.tsv w]
    puts $pf "name\tlevel\tslew_r\tslew_f\tat_r\tat_f\tnet\tcap_ff\twire_ff"
    set af [open $directory/arcs.tsv w]
    puts $af "src\tdst\trole\tsense\twhen\tfr\tto\tdelay_ps\tdisabled\telmore_ps"
    set it [sta::vertex_iterator]
    while {[$it has_next]} {
        set v [$it next];set pin [$v pin];set values [list [get_full_name $pin] [$v level]]
        foreach rf {rise fall} {
            set text [$v slew_scenes_string $rf [list $ptr] $mm false 12]
            lappend values [expr {[sta::time_ui_sta $text]*1e12}]
        }
        foreach rf {rise fall} {lappend values [analysis_graph_arrival $v $scene $rf $mm $single_plain_scene]}
        set net [$pin net]
        if {$net eq "NULL"} {set term [$pin term];if {$term ne "NULL"} {set net [$term net]}}
        if {$net eq "NULL"} {lappend values "" 0 0} else {
            lappend values [get_full_name $net] [expr {[$net capacitance $ptr $mm]*1e15}] [expr {[$net wire_capacitance $ptr $mm]*1e15}]
        }
        puts $pf [join $values "\t"]
        set ei [$v out_edge_iterator]
        while {[$ei has_next]} {
            set edge [$ei next];set disabled 0
            foreach method {is_disabled_loop is_disabled_constraint is_disabled_constant is_disabled_cond_default is_disabled_bidirect_inst_path is_disabled_preset_clear} {
                if {[$edge $method]} {set disabled 1}
            }
            foreach arc [$edge timing_arcs] {
                set fr [$arc from_edge_name];set to [$arc to_edge_name];set elmore 0
                if {[$edge role] eq "wire"} {
                    set elmore [expr {[sta::find_elmore [$edge from_pin] [$edge to_pin] $fr $mm]*1e12}]
                }
                puts $af [join [list [get_full_name [$edge from_pin]] [get_full_name [$edge to_pin]] \
                    [$edge role] [$edge sim_timing_sense] [$edge cond] $fr $to \
                    [expr {[$edge arc_delay $arc $ptr $mm]*1e12}] $disabled $elmore] "\t"]
            }
        }
        $ei finish
    }
    $it finish;close $pf;close $af
    set ef [open $directory/endpoints.tsv w]
    puts $ef "pin\trf\trole\tat_ps\trequired_ps\tslack_ps\tmargin_ps\tcapture_pin\tcapture_rf\tcapture_at_ps\tstartpoint"
    foreach rf {rise fall} {
        foreach end [find_timing_paths -scenes $scene -path_delay $mm -${rf}_to [sta::endpoints] -group_path_count 100000 -endpoint_path_count 1] {
            set transition [$end end_transition]
            if {$transition eq "^"} {set transition rise}
            if {$transition eq "v"} {set transition fall}
            set role [$end check_role]
            if {![$end is_check] && ([$end is_output_delay] || [$end is_path_delay])} {set role "output $role"}
            set capture_pin "";set capture_rf "";set capture_at 0
            set clock_path [$end target_clk_path]
            if {$clock_path ne "NULL"} {
                set capture_pin [get_full_name [$clock_path pin]];set capture_rf [$clock_path edge]
                if {$capture_rf eq "^"} {set capture_rf rise}
                if {$capture_rf eq "v"} {set capture_rf fall}
                set capture_at [expr {[$clock_path arrival]*1e12}]
            }
            puts $ef [join [list [get_full_name [$end pin]] $transition $role \
                [expr {[$end data_arrival_time]*1e12}] [expr {[$end data_required_time]*1e12}] \
                [expr {[$end slack]*1e12}] [expr {[$end margin]*1e12}] \
                $capture_pin $capture_rf $capture_at [get_full_name [get_property $end startpoint]]] "\t"]
        }
    }
    close $ef
    set mf [open $directory/metrics.tsv w]
    puts $mf "tns_ps\twns_ps"
    puts $mf "[expr {[sta::total_negative_slack_scene_cmd $ptr $mm]*1e12}]\t[expr {[sta::worst_slack_scene $ptr $mm]*1e12}]"
    close $mf
}
