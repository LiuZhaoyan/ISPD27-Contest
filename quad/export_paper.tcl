# Reference quantities for paper-model validation. Does not alter constraints.
source [file join [file dirname [info script]] export_stage48.tcl]
proc paper_export {directory {spef ""}} {
    file mkdir $directory
    set scene [quad_scene_name]
    if {$spef ne ""} {
        file copy -force $spef $directory/network.spef
    } else {
        estimate_parasitics -global_routing -spef_file $directory/network.spef
        file copy -force $directory/network_${scene}.spef $directory/network.spef
    }
    if {![file exists $directory/network.spef]} {error "Selected scene SPEF missing: $scene"}
    stage48_export $directory
    sta::unset_power_input_activity
    sta::design_power [sta::cmd_scene]
    set f [open $directory/activity.tsv w]
    puts $f "pin\tdensity_hz\tduty\torigin"
    set it [sta::vertex_iterator]
    while {[$it has_next]} {
        set v [$it next]; set p [$v pin]
        lassign [get_property $p activity] density duty origin
        puts $f [join [list [get_full_name $p] $density $duty $origin] "\t"]
    }
    $it finish
    close $f
    set f [open $directory/power.tsv w]
    puts $f "instance\tinternal_w\tswitching_w\tleakage_w\ttotal_w"
    foreach i [get_cells -hierarchical *] {
        puts $f [join [linsert [sta::instance_power $i [sta::cmd_scene]] 0 [get_full_name $i]] "\t"]
    }
    close $f
}
