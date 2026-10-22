# Separate OpenROAD process. No evaluator source is altered.
set root $::env(QG_ROOT)
set baseline $::env(QG_BASELINE)
set output $::env(QG_OUTPUT)
set initialize [expr {[info exists ::env(QG_INITIALIZE)] && $::env(QG_INITIALIZE)}]
set route [expr {[info exists ::env(QG_ROUTE)] && $::env(QG_ROUTE)}]
set modifying [expr {[info exists ::env(QG_ACTION)] && $::env(QG_ACTION) ne ""}]
source [file join $root evaluator internal legality_snapshot.tcl]
set ::env(PLATFORM_DIR) [file join $root platform asap7]
source $::env(MCMM_CONFIG)
set_thread_count 8
source [file join $::env(PLATFORM_DIR) util liberty_suppressions.tcl]
source [file join $::env(PLATFORM_DIR) util libraries.tcl]
foreach lef {asap7_tech_1x_201209.lef asap7sc7p5t_28_R_1x_220121a.lef asap7sc7p5t_28_L_1x_220121a.lef asap7sc7p5t_28_SL_1x_220121a.lef} {
    read_lef [file join $::env(PLATFORM_DIR) lef $lef]
}
foreach scene {TC BC WC} { foreach lib $corner_libs($scene) { read_liberty $lib } }
if {$initialize} { read_def [file join $baseline input.def] } else {
    read_def [file join $baseline evaluated.def]
}
foreach scene {TC BC WC} {
    set mode $scenario_mode($scene)
    read_sdc -mode $mode $scenario_sdc($scene)
    set_mode $mode
    foreach clk [all_clocks] {
        if {![get_property $clk is_virtual]} { set_propagated_clock $clk }
    }
    if {$initialize || $route || $modifying} {
        define_scene $scene -mode $mode -liberty $corner_libs($scene)
    } else {
        read_spef -name rc_$scene [file join $baseline parasitics_${scene}.spef]
        define_scene $scene -mode $mode -liberty $corner_libs($scene) -spef rc_$scene
        report_parasitic_annotation -name rc_$scene -report_unannotated > [file join $output ${scene}_parasitic_annotation.rpt]
    }
}
set_scene TC
source [file join $::env(PLATFORM_DIR) util setRC.tcl]
legality_snapshot [file join $output before] 1
# DEF does not preserve dont_touch: restore protection on every load.
foreach net [[ord::get_db_block] getNets] {
        if {[$net getSigType] ne "CLOCK"} { continue }
        $net setDoNotTouch true
        foreach pin [$net getITerms] {
            set inst [$pin getInst]
            $inst setPlacementStatus FIRM
            $inst setDoNotTouch true
        }
}
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
# Apply before routing in the same disposable process; neither stage loads old SPEFs.
if {$modifying} {
    set f [open $::env(QG_ACTION) r]
    set replacements [dict create]
    set legalize 0
    set geometry [table geometry.tsv {instance before_width before_height after_width after_height legalized}]
    while {[gets $f line] >= 0} {
        if {[llength [split $line \t]] != 2} { error "Malformed batch action" }
        lassign [split $line \t] name target
        if {$name eq "" || $target eq "" || [dict exists $replacements $name]} {
            error "Empty or duplicate batch instance"
        }
        set inst [[ord::get_db_block] findInst $name]
        set target_master [[ord::get_db] findMaster $target]
        if {$inst eq "NULL" || $target_master eq "NULL"} { error "Missing batch instance or master" }
        if {[$inst isDoNotTouch] || [$inst getPlacementStatus] in {FIRM LOCKED}} {
            error "Protected batch instance: $name"
        }
        set old_master [$inst getMaster]
        set moved [expr {[$old_master getWidth] != [$target_master getWidth] || [$old_master getHeight] != [$target_master getHeight]}]
        if {$moved} { set legalize 1 }
        row $geometry $name [$old_master getWidth] [$old_master getHeight] [$target_master getWidth] [$target_master getHeight] $moved
        dict set replacements $name $target
    }
    close $f
    close $geometry
    # One transaction in a disposable process, including best-state freeze restores.
    dict for {name target} $replacements {
        replace_cell [get_cells $name] $target
        set inst [[ord::get_db_block] findInst $name]
        if {[[$inst getMaster] getName] ne $target || [get_property [get_cells $name] ref_name] ne $target} {
            error "STA/OpenDB replacement mismatch"
        }
    }
    if {$legalize} { detailed_placement }
}
if {$initialize || $route} {
    if {[[ord::get_db_block] getName] ne $::env(QG_TOP)} { error "Top module mismatch" }
    set_dont_use {*x1p*_ASAP7* *xp*_ASAP7* SDF* ICG*}
    check_placement -verbose
    # B is both apply-after and route-before. Keep evaluated.def/v pre-route,
    # and export the library catalog only with A (before).
    if {$modifying} {
        legality_snapshot [file join $output evaluated]
    } else {
        write_def [file join $output evaluated.def]
    }
    write_verilog [file join $output evaluated.v]
    # Same physical setup as the public R0, but no proofs, full metric extraction
    # or clean verification subprocesses. Initial STA is the subsequent probe.
    set_routing_layers -signal M2-M7 -clock M4-M7
    set_global_routing_layer_adjustment * 0.25
    set_global_routing_random -seed 42
    global_route -allow_congestion -congestion_iterations 30 \
        -congestion_report_file [file join $output congestion.rpt]
    estimate_parasitics -global_routing -spef_file [file join $output parasitics.spef]
    legality_snapshot [file join $output snapshot]
    # Preserve per-layer overflow, without offsetting it by spare capacity.
    set grid [[ord::get_db_block] getGCellGrid]
    if {$grid eq "NULL" || ![llength [$grid getGridX]] || ![llength [$grid getGridY]]} {
        error "Missing routing grid"
    }
    set total 0; set peak 0
    foreach layer [[ord::get_db_tech] getLayers] {
        if {[$layer getRoutingLevel] <= 0} { continue }
        for {set x 0} {$x < [llength [$grid getGridX]]} {incr x} {
            for {set y 0} {$y < [llength [$grid getGridY]]} {incr y} {
                set over [expr {max(0,[$grid getUsage $layer $x $y]-[$grid getCapacity $layer $x $y])}]
                incr total $over
                set peak [expr {max($peak,$over)}]
            }
        }
    }
    set fp [open [file join $output overflow.tsv] w]
    puts $fp "total\tmax"
    puts $fp "$total\t$peak"
    close $fp
    puts QG_INITIALIZATION_COMPLETE
    exit
}
legality_snapshot [file join $output snapshot] 1
write_verilog [file join $output loaded.v]
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
