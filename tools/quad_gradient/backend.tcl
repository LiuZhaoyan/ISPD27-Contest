# Separate OpenROAD process. No evaluator source is altered.
set root $::env(QG_ROOT)
set baseline $::env(QG_BASELINE)
set output $::env(QG_OUTPUT)
set ::env(PLATFORM_DIR) [file join $root platform asap7]
source $::env(MCMM_CONFIG)
set_thread_count 8
source [file join $::env(PLATFORM_DIR) util liberty_suppressions.tcl]
source [file join $::env(PLATFORM_DIR) util libraries.tcl]
foreach lef {asap7_tech_1x_201209.lef asap7sc7p5t_28_R_1x_220121a.lef asap7sc7p5t_28_L_1x_220121a.lef asap7sc7p5t_28_SL_1x_220121a.lef} {
    read_lef [file join $::env(PLATFORM_DIR) lef $lef]
}
foreach scene {TC BC WC} { foreach lib $corner_libs($scene) { read_liberty $lib } }
read_def [file join $baseline evaluated.def]
foreach scene {TC BC WC} {
    set mode $scenario_mode($scene)
    read_sdc -mode $mode $scenario_sdc($scene)
    set_mode $mode
    foreach clk [all_clocks] {
        if {![get_property $clk is_virtual]} { set_propagated_clock $clk }
    }
    read_spef -name rc_$scene [file join $baseline parasitics_${scene}.spef]
    define_scene $scene -mode $mode -liberty $corner_libs($scene) -spef rc_$scene
}
set_scene TC
source [file join $::env(PLATFORM_DIR) util setRC.tcl]
source [file join $root evaluator internal legality_snapshot.tcl]
proc row {fp args} { puts $fp [join $args \t] }
proc table {name fields} {
    set f [open [file join $::output $name] w]
    puts $f [join $fields \t]
    return $f
}
proc fullname {object} {
    if {$object eq "NULL" || $object eq ""} { return "" }
    return [get_full_name $object]
}
proc rf {edge} {
    if {$edge eq "^"} {return rise}
    if {$edge eq "v"} {return fall}
    return $edge
}
if {[info exists ::env(QG_ACTION)] && $::env(QG_ACTION) ne ""} {
    set f [open $::env(QG_ACTION) r]
    lassign [split [string trim [read $f]] \t] name target
    close $f
    set inst [[ord::get_db_block] findInst $name]
    set old_master [$inst getMaster]
    set target_master [[ord::get_db] findMaster $target]
    if {$target_master eq "NULL"} { error "Missing target OpenDB master" }
    set geometry [table geometry.tsv {instance before_width before_height after_width after_height legalized}]
    row $geometry $name [$old_master getWidth] [$old_master getHeight] [$target_master getWidth] [$target_master getHeight] $::env(QG_LEGALIZE)
    close $geometry
    if {!$::env(QG_LEGALIZE) && ([$old_master getWidth] != [$target_master getWidth] || [$old_master getHeight] != [$target_master getHeight])} {
        error "VT footprint mismatch"
    }
    replace_cell [get_cells $name] $target
    set inst [[ord::get_db_block] findInst $name]
    if {[[$inst getMaster] getName] ne $target || [get_property [get_cells $name] ref_name] ne $target} {
        error "STA/OpenDB replacement mismatch"
    }
    # Same-footprint VT needs no motion; drive trials always legalize.
    if {$::env(QG_LEGALIZE)} { detailed_placement }
    check_placement -verbose
    write_def [file join $output candidate.def]
    write_verilog [file join $output candidate.v]
}
legality_snapshot [file join $output snapshot] 1
set views [table views.tsv {master view}]
foreach scene {TC BC WC} {
    foreach lib $corner_libs($scene) {
        # Actual library cells, not filename-based cell-name inference.
        foreach cell [get_lib_cells -quiet *] {
            set libname [[$cell liberty_library] name]
            if {[string first $libname [file tail $lib]] >= 0} { row $views [$cell name] $scene }
        }
    }
}
foreach lib [[ord::get_db] getLibs] {
    foreach master [$lib getMasters] { row $views [$master getName] LEF }
}
close $views
set native [table native.tsv {scene analysis wns_ps tns_ps}]
foreach scene {TC BC WC} {
    foreach analysis {max min} {
        row $native $scene $analysis [sta::worst_slack -scene $scene -$analysis] [sta::total_negative_slack -scene $scene -$analysis]
    }
    foreach type {max_slew max_capacitance max_fanout} {
        report_check_types -scenes $scene -$type -violators -digits 9 -no_line_splits > [file join $output ${scene}_${type}.rpt]
    }
}
close $native
if {![info exists ::env(QG_EXPORT_GRAPH)] || !$::env(QG_EXPORT_GRAPH)} { exit }
set_scene TC
sta::find_requireds
set scene_obj [sta::find_scene TC]
set points [table paths.tsv {path endpoint slack_ps index pin transition arrival_ps slew_ps tag}]
set samples [table arcs.tsv {kind sample_id scene mode from_pin to_pin role sense arc_ordinal from_transition to_transition when liberty_cell graph_delay_s report_file}]
set seen [dict create]
set n 0
foreach path [find_timing_paths -scenes TC -path_delay max -group_path_count 1000000 -endpoint_path_count 2 -slack_max 1e30] {
    set endpoint [fullname [get_property $path endpoint]]
    set i 0
    foreach point [get_property $path points] {
        set pin [get_property $point pin]
        set trans [rf [$point edge]]
        set slews {}
        foreach v [$pin vertices] { lappend slews [$v slew_scenes_string $trans [list $scene_obj] max false 9] }
        row $points $n $endpoint [get_property $path slack] $i [fullname $pin] $trans [get_property $point arrival] [lindex $slews 0] [$point tag]
        dict set seen [fullname $pin] $pin
        incr i
    }
    incr n
}
close $points
set n 0
# Include competing input arcs for every observed data output. Native checks bind
# the setup seed. Structural/tag compatibility remains an explicit approximation.
dict for {name pin} $seen {
    foreach edge [get_timing_edges -to $pin] {
        set role [$edge role]
        if {$role ni {combinational setup {Reg Clk to Q}}} {continue}
        if {[$edge is_disabled_loop] || [$edge is_disabled_constraint] || [$edge is_disabled_constant] || [$edge is_disabled_cond_default]} {continue}
        set inst [$pin instance]
        foreach arc [$edge timing_arcs] {
            set kind [expr {$role eq "setup" ? "setup" : "load"}]
            set id arc_$n
            set file ${id}.rpt
            set f [open [file join $output $file] w]
            puts $f [sta::report_delay_calc_cmd $edge $arc $scene_obj max 9]
            close $f
            row $samples $kind $id TC $scenario_mode(TC) [fullname [$edge from_pin]] [fullname $pin] $role [$edge sense] $n \
                [$arc from_edge_name] [$arc to_edge_name] [$edge cond] [get_property [get_cells -of_objects $pin] ref_name] \
                [$edge arc_delay $arc $scene_obj max] $file
            incr n
        }
    }
}
close $samples
puts QG_EXPORT_COMPLETE
