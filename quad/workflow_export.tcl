# Called in a single-scene worker after loading the independent GR SPEF.
proc workflow_endpoint_export {directory scene analysis} {
    if {$analysis ni {min max}} {error "Expected min/max endpoints"}
    file mkdir $directory
    set_scene $scene
    sta::find_requireds
    set f [open $directory/endpoints.tsv w]
    puts $f "pin\trf\tslack_ps"
    foreach rf {rise fall} {
        foreach end [find_timing_paths -scenes $scene -path_delay $analysis -${rf}_to [sta::endpoints] -group_path_count 100000 -endpoint_path_count 1] {
            set edge [$end end_transition]
            if {$edge eq "^"} {set edge rise}
            if {$edge eq "v"} {set edge fall}
            puts $f [join [list [get_full_name [$end pin]] $edge [expr {[$end slack]*1e12}]] "\t"]
        }
    }
    close $f
}

proc workflow_export {directory scene placement_spef} {
    foreach clock [all_clocks] {
        if {[get_property $clock is_generated]} {error "Workflow requires explicit generated-clock tag support"}
    }
    foreach analysis {max min} {
        analysis_graph_export $directory/gr/$analysis $scene $analysis
        source_group_export $directory/groups/$analysis $scene $analysis
    }
    # Power/activity are GR timing annotations, never a physical RC input.
    sta::unset_power_input_activity
    sta::design_power [sta::cmd_scene]
    set f [open $directory/activity.tsv w]
    puts $f "pin\tdensity_hz\tduty\torigin"
    set it [sta::vertex_iterator]
    while {[$it has_next]} {
        set p [[$it next] pin]
        lassign [get_property $p activity] density duty origin
        puts $f [join [list [get_full_name $p] $density $duty $origin] "\t"]
    }
    $it finish; close $f
    # Native placement parasitics replace the worker's named network only after
    # all GR reference data have been exported. The parent remains a GR oracle.
    read_spef -name quad_rc $placement_spef
    foreach analysis {max min} {
        analysis_graph_export $directory/placement/$analysis $scene $analysis
        file copy -force $placement_spef $directory/placement/$analysis/network.spef
        file copy -force $directory/activity.tsv $directory/placement/$analysis/activity.tsv
    }
}
