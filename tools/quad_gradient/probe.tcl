# Read-only, separate process: never source or modify the evaluator execution.
set root $::env(QG_ROOT)
set baseline $::env(QG_BASELINE)
set output $::env(QG_OUTPUT)
set ::env(PLATFORM_DIR) [file join $root platform asap7]
source [file join $root benchmarks aes mcmm.tcl]
set_thread_count 8
source [file join $::env(PLATFORM_DIR) util liberty_suppressions.tcl]
source [file join $::env(PLATFORM_DIR) util libraries.tcl]
foreach lef {asap7_tech_1x_201209.lef asap7sc7p5t_28_R_1x_220121a.lef asap7sc7p5t_28_L_1x_220121a.lef asap7sc7p5t_28_SL_1x_220121a.lef} {
    read_lef [file join $::env(PLATFORM_DIR) lef $lef]
}
foreach scene {TC BC WC} {
    foreach lib $corner_libs($scene) { read_liberty $lib }
}
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
report_units
set units [open [file join $output units.rpt] w]
foreach unit {time capacitance resistance voltage current power distance} {
    puts $units "$unit [sta::unit_scale_suffix $unit]"
}
close $units
# TSV fields escape control characters, retaining backslashes reversibly.
proc row {fp args} {
    set fields {}
    foreach value $args {
        lappend fields [string map {\\ \\\\ \t \\t \n \\n \r \\r} $value]
    }
    puts $fp [join $fields \t]
}
proc table {name fields} {
    set fp [open [file join $::output $name] w]
    puts $fp [join $fields \t]
    return $fp
}
proc transition {rf} {
    if {$rf eq "^"} { return rise }
    if {$rf eq "v"} { return fall }
    return $rf
}
proc fullname {object} {
    if {$object eq "NULL" || $object eq ""} { return "" }
    return [get_full_name $object]
}
set native [table native.tsv {scene analysis wns_ps tns_ps}]
set endpoints [table endpoints.tsv {scene analysis endpoint slack_ps transition startpoint launch_clock capture_clock check_role margin_s required_s arrival_s}]
set points [table points.tsv {scene mode analysis path_id index pin transition tag arrival_ps required_ps slack_ps slew_ps net total_cap_F pin_cap_F wire_cap_F}]
set masters [table masters.tsv {instance sta_master odb_master}]
set caps [table capabilities.tsv {capability status detail}]
set selected {}
foreach scene {TC BC WC} {
    set_scene $scene
    set scene_obj [sta::find_scene $scene]
    foreach analysis {max min} {
        row $native $scene $analysis [sta::worst_slack -scene $scene -$analysis] [sta::total_negative_slack -scene $scene -$analysis]
        foreach path [find_timing_paths -scenes $scene -path_delay $analysis -group_path_count 1000000 -endpoint_path_count 1 -slack_max 1e30] {
            row $endpoints $scene $analysis [fullname [get_property $path endpoint]] [get_property $path slack] \
                [transition [$path end_transition]] [fullname [get_property $path startpoint]] \
                [fullname [get_property $path startpoint_clock]] [fullname [$path target_clk]] \
                [$path check_role] [$path margin] [$path data_required_time] [$path data_arrival_time]
        }
        report_checks -scenes $scene -path_delay $analysis -group_path_count 1 -endpoint_path_count 1 \
            -format full_clock_expanded -fields {slew capacitance fanout} -digits 9 > [file join $output ${scene}_${analysis}_path.rpt]
        set id 0
        foreach path [find_timing_paths -scenes $scene -path_delay $analysis -group_path_count 1 -endpoint_path_count 1] {
            set i 0
            foreach point [get_property $path points] {
                set pin [get_property $point pin]
                set rf [transition [$point edge]]
                set net [$pin net]
                if {$net eq "NULL" && [$pin is_top_level_port]} {
                    set term [$pin term]
                    if {$term ne "NULL"} { set net [$term net] }
                }
                set netname ""; set total ""; set pin_cap ""; set wire ""
                if {$net ne "NULL" && $net ne ""} {
                    set netname [fullname $net]
                    set total [$net capacitance $scene_obj $analysis]
                    set pin_cap [$net pin_capacitance $scene_obj $analysis]
                    set wire [$net wire_capacitance $scene_obj $analysis]
                }
                set slew {}
                foreach vertex [$pin vertices] {
                    lappend slew [$vertex slew_scenes_string $rf [list $scene_obj] $analysis false 9]
                }
                row $points $scene $scenario_mode($scene) $analysis $id $i [fullname $pin] $rf [$point tag] \
                    [get_property $point arrival] [get_property $point required] [get_property $point slack] \
                    $slew $netname $total $pin_cap $wire
                if {$scene eq "TC" && $analysis eq "max"} { lappend selected $pin }
                incr i
            }
            incr id
        }
    }
}
close $native; close $endpoints; close $points
foreach inst [[ord::get_db_block] getInsts] {
    set name [$inst getName]
    set sta_inst [get_cells -quiet $name]
    if {[llength $sta_inst] == 1} { row $masters $name [get_property $sta_inst ref_name] [[$inst getMaster] getName] }
}
close $masters
# Bounded structural fan-in around the selected TC max path. Every incoming
# edge is recorded; no edge is claimed to connect compatible timing tags.
set edges [table fanin.tsv {to_pin from_pin role sense disabled_loop arc_id from_transition to_transition}]
set states [table states.tsv {scene mode pin analysis transition ordinal tag arrival_s required_s truncated}]
set arcs [table liberty_arcs.tsv {instance cell arc_set_id from to role when from_transition to_transition}]
set seen [dict create]
foreach pin $selected {
    foreach edge [get_timing_edges -to $pin] {
        set from [$edge from_pin]
        foreach p [list $pin $from] { dict set seen [fullname $p] $p }
        set n 0
        foreach arc [$edge timing_arcs] {
            row $edges [fullname $pin] [fullname $from] [$edge role] [$edge sense] [$edge is_disabled_loop] $n [$arc from_edge_name] [$arc to_edge_name]
            incr n
        }
    }
}
close $edges
# Object identities are local ordinals, not persistent SWIG pointers.
set cells [dict create]
dict for {name pin} $seen {
    foreach inst [get_cells -quiet -of_objects $pin] { dict set cells [fullname $inst] $inst }
    foreach vertex [$pin vertices] {
        foreach analysis {max min} {
            foreach rf {rise fall} {
                set iterator [$vertex path_iterator $rf $analysis]
                set n 0
                while {[$iterator has_next] && $n < 32} {
                    set path [$iterator next]
                    # Pinned backend Tag::to_string prefixes index + scene.
                    # Reject an unknown layout rather than assume current scene.
                    set tag [$path tag]
                    if {![regexp {^\S+ (BC|TC|WC) } $tag unused state_scene]} {
                        error "Unrecognized scene in vertex timing tag: $tag"
                    }
                    row $states $state_scene $scenario_mode($state_scene) $name $analysis $rf $n $tag [$path arrival] [$path required] 0
                    incr n
                }
                if {[$iterator has_next]} { row $states unknown unknown $name $analysis $rf $n "" "" "" 1 }
                $iterator finish
            }
        }
    }
}
close $states
dict for {name inst} $cells {
    set cell [get_property $inst liberty_cell]
    set n 0
    foreach arcset [$cell timing_arc_sets] {
        foreach arc [$arcset timing_arcs] {
            row $arcs $name [fullname $cell] $n [fullname [$arcset from]] [fullname [$arcset to]] \
                [$arcset role] [$arcset when] [$arc from_edge_name] [$arc to_edge_name]
        }
        incr n
    }
}
close $arcs
row $caps explicit_scene_slew observed "Vertex slew_scenes_string on selected scene paths; ps report units"
row $caps connected_capacitance observed "Explicit scene/minmax pin, wire and total; SI farads; not effective load"
row $caps vertex_tag_enumeration observed "At most 32 states per vertex/transition/analysis; truncation flagged"
row $caps tagged_predecessors unverified "No legal tag-to-tag edge/check mapping established"
row $caps setup_lut_binding unverified "Check role/margin and library arcs exported; selected table binding not established"
close $caps
if {[info exists ::env(QG_CLOSURE_SCENE)]} {
    # Additional read-only native references, with a preregistered sample manifest.
    set_scene $::env(QG_CLOSURE_SCENE)
    set scene_obj [sta::find_scene $::env(QG_CLOSURE_SCENE)]
    set analysis max
    sta::find_requireds
    set interfaces [open [file join $output closure_interfaces.txt] w]
    foreach pattern {sta::*Path* sta::*Tag* sta::*required* sta::*search* sta::*compatible* sta::*report*d*calc*} {
        puts $interfaces "$pattern: [lsort [info commands $pattern]]"
    }
    close $interfaces
    set boundary [table closure_boundary.tsv {pin boundary_role}]
    set relations [table closure_relations.tsv {to_pin from_pin role sense disabled_loop disabled_constraint disabled_constant disabled_cond_default arc_ordinal from_transition to_transition when}]
    set rawstates [table closure_states.tsv {scene mode pin analysis transition ordinal tag arrival_s required_s constrained_validity truncated}]
    set pins [dict create]
    set target [sta::get_port_pin_error fanin_output $::env(QG_CLOSURE_FANIN)]
    dict set pins [fullname $target] $target
    row $boundary [fullname $target] output
    foreach edge [get_timing_edges -to $target] {
        set from [$edge from_pin]
        dict set pins [fullname $from] $from
        row $boundary [fullname $from] cell_input
        set edges [list $edge]
        foreach incoming [get_timing_edges -to $from] {
            if {[$incoming role] eq "wire"} {
                lappend edges $incoming
                set driver [$incoming from_pin]
                dict set pins [fullname $driver] $driver
                row $boundary [fullname $driver] wire_boundary_driver
            }
        }
        foreach e $edges {
            set n 0
            foreach arc [$e timing_arcs] {
                row $relations [fullname [$e to_pin]] [fullname [$e from_pin]] [$e role] [$e sense] \
                    [$e is_disabled_loop] [$e is_disabled_constraint] [$e is_disabled_constant] [$e is_disabled_cond_default] \
                    $n [$arc from_edge_name] [$arc to_edge_name] [$e cond]
                incr n
            }
        }
    }
    close $boundary; close $relations
    dict for {name pin} $pins {
        puts "QG_NATIVE_STATE_BEGIN $name"
        report_arrival -scene $::env(QG_CLOSURE_SCENE) -digits 9 $pin
        report_required -scene $::env(QG_CLOSURE_SCENE) -digits 9 $pin
        puts "QG_NATIVE_STATE_END $name"
        foreach vertex [$pin vertices] {
            foreach rf {rise fall} {
                set iterator [$vertex path_iterator $rf max]
                set n 0
                while {[$iterator has_next] && $n < $::env(QG_CLOSURE_LIMIT)} {
                    set path [$iterator next]
                    set tag [$path tag]
                    if {![regexp {^\S+ (BC|TC|WC) } $tag unused state_scene]} { error "Unknown tag format: $tag" }
                    # Iterator includes all scenes; filter by native tag's actual scene.
                    if {$state_scene eq $::env(QG_CLOSURE_SCENE)} {
                        row $rawstates $state_scene $scenario_mode($state_scene) $name max $rf $n $tag \
                            [$path arrival] [$path required] unverified 0
                    }
                    incr n
                }
                if {[$iterator has_next]} {
                    row $rawstates unknown unknown $name max $rf $n "" "" "" unknown 1
                }
                $iterator finish
            }
        }
    }
    close $rawstates
    set samples [table closure_samples.tsv {kind sample_id scene mode from_pin to_pin role sense arc_ordinal from_transition to_transition when liberty_cell native_delay_s report_file}]
    set endpoints [table closure_endpoints.tsv {sample_id endpoint data_transition check_role launch_clock capture_clock margin_s arrival_s required_s slack_ps unconstrained}]
    # Numbered environment keys preserve literal HDL pin names without evaluating Tcl.
    for {set sample 0} {$sample < $::env(QG_CLOSURE_ENDPOINTS)} {incr sample} {
        set pin [sta::get_port_pin_error endpoint $::env(QG_CLOSURE_ENDPOINT_$sample)]
        set inst [$pin instance]
        set clkpin [sta::get_port_pin_error clock_pin "[fullname $inst]/CLK"]
        foreach rf {rise fall} {
            set paths [find_timing_paths -scenes $::env(QG_CLOSURE_SCENE) -path_delay max -${rf}_to $pin -group_path_count 1 -endpoint_path_count 1]
            foreach path $paths {
                set id endpoint_${sample}_$rf
                row $endpoints $id [fullname $pin] [transition [$path end_transition]] [$path check_role] \
                    [fullname [get_property $path startpoint_clock]] [fullname [$path target_clk]] \
                    [$path margin] [$path data_arrival_time] [$path data_required_time] [get_property $path slack] [$path is_unconstrained]
                report_checks -scenes $::env(QG_CLOSURE_SCENE) -path_delay max -${rf}_to $pin -group_path_count 1 \
                    -endpoint_path_count 1 -format full_clock_expanded -fields {slew capacitance} -digits 9 > [file join $output ${id}_path.rpt]
            }
        }
        foreach edge [get_timing_edges -from $clkpin -to $pin] {
            if {[$edge role] ne "setup"} { continue }
            set n 0
            foreach arc [$edge timing_arcs] {
                set id setup_${sample}_$n
                set file ${id}_dcalc.rpt
                set fp [open [file join $output $file] w]
                puts $fp [sta::report_delay_calc_cmd $edge $arc $scene_obj max 9]
                close $fp
                # Native graph check delay and raw delay calculator value are different references.
                row $samples setup $id $::env(QG_CLOSURE_SCENE) $scenario_mode($::env(QG_CLOSURE_SCENE)) \
                    [fullname $clkpin] [fullname $pin] [$edge role] [$edge sense] $n \
                    [$arc from_edge_name] [$arc to_edge_name] [$edge cond] [get_property $inst ref_name] \
                    [$edge arc_delay $arc $scene_obj max] $file
                incr n
            }
        }
    }
    close $endpoints
    set loads [table closure_load_points.tsv {driver scene transition total_cap_F pin_cap_F wire_cap_F output_slew_ps report_file}]
    set net_slews [table closure_net_slews.tsv {driver pin scene transition slew_ps}]
    for {set sample 0} {$sample < $::env(QG_CLOSURE_DRIVERS)} {incr sample} {
        set pin [sta::get_port_pin_error driver $::env(QG_CLOSURE_DRIVER_$sample)]
        set inst [$pin instance]
        set net [get_nets -quiet -of_objects $pin]
        if {[llength $net] != 1} { error "Expected one connected driver net" }
        set net [lindex $net 0]
        foreach rf {rise fall} {
            set report_file driver_${sample}_${rf}_path.rpt
            report_checks -scenes $::env(QG_CLOSURE_SCENE) -path_delay max -${rf}_through $pin \
                -group_path_count 1 -endpoint_path_count 1 -format full_clock_expanded \
                -fields {slew capacitance fanout} -digits 9 > [file join $output $report_file]
            foreach vertex [$pin vertices] {
                row $loads [fullname $pin] $::env(QG_CLOSURE_SCENE) $rf [$net capacitance $scene_obj max] \
                    [$net pin_capacitance $scene_obj max] [$net wire_capacitance $scene_obj max] \
                    [$vertex slew_scenes_string $rf [list $scene_obj] max false 9] $report_file
            }
            foreach connected [get_pins -quiet -of_objects $net] {
                foreach vertex [$connected vertices] {
                    row $net_slews [fullname $pin] [fullname $connected] $::env(QG_CLOSURE_SCENE) $rf \
                        [$vertex slew_scenes_string $rf [list $scene_obj] max false 9]
                }
            }
        }
        set edge_n 0
        foreach edge [get_timing_edges -to $pin] {
            if {[$edge role] ne "combinational"} { continue }
            set n 0
            foreach arc [$edge timing_arcs] {
                set id load_${sample}_${edge_n}_$n
                set file ${id}_dcalc.rpt
                set fp [open [file join $output $file] w]
                puts $fp [sta::report_delay_calc_cmd $edge $arc $scene_obj max 9]
                close $fp
                row $samples load $id $::env(QG_CLOSURE_SCENE) $scenario_mode($::env(QG_CLOSURE_SCENE)) \
                    [fullname [$edge from_pin]] [fullname $pin] [$edge role] [$edge sense] $n \
                    [$arc from_edge_name] [$arc to_edge_name] [$edge cond] [get_property $inst ref_name] \
                    [$edge arc_delay $arc $scene_obj max] $file
                incr n
            }
            incr edge_n
        }
    }
    close $loads; close $net_slews; close $samples
    puts "QG_M0_CLOSURE_EXPORT_COMPLETE"
}
puts "QG_M0_EXPORT_COMPLETE"
