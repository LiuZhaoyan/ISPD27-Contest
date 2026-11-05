# Source after loading constraints and estimating parasitics. All numerical
# quantities below are native SI, converted explicitly to ps and fF.
proc quad_scene_name {} {
    set scene [sta::cmd_scene]
    # The older local build returns a name; the contest build returns a
    # Scene object. Both interfaces were checked against actual binaries.
    if {$scene in {BC TC WC}} {return $scene}
    return [get_property $scene name]
}
proc quad_pin_arrivals {vertex scene} {
    set arrivals [list -Inf -Inf]
    set single [expr {[llength [sta::scenes]]==1}]
    set i 0
    foreach rf {rise fall} {
        set paths [$vertex path_iterator $rf max]
        while {[$paths has_next]} {
            set path [$paths next]
            if {$single && !$::quad_has_genclk} {
                lset arrivals $i [expr {max([lindex $arrivals $i],[$path arrival]*1e12)}]
                continue
            }
            set tag [$path tag]
            if {![regexp {^([0-9]+) ([^ ]+) } $tag -> index tag_scene]} {error "Unsupported STA tag format"}
            if {$tag_scene eq $scene && ![string match *genclk* $tag]} {
                lset arrivals $i [expr {max([lindex $arrivals $i],[$path arrival]*1e12)}]
            }
        }
        $paths finish; incr i
    }
    return $arrivals
}
proc stage48_export {directory} {
    file mkdir $directory
    sta::find_requireds
    set scene [quad_scene_name]
    set scene_ptr [sta::cmd_scene]
    if {$scene ni {BC TC WC}} {error "Unsupported export scene name: $scene"}
    set finite_arrivals 0
    set ::quad_has_genclk 0
    foreach clk [all_clocks] {if {[get_property $clk is_generated]} {set ::quad_has_genclk 1}}
    set pin_count 0
    puts "QUAD_EXPORT_PINS $scene"; flush stdout
    set pf [open $directory/pins.tsv w]
    puts $pf "name\tlevel\tslew_r\tslew_f\tat_r\tat_f\tnet\tcap_ff\twire_ff"
    set ef [open $directory/arcs.tsv w]
    puts $ef "src\tdst\trole\tsense\twhen\tfr\tto\tdelay_ps\tdisabled\telmore_ps"
    set it [sta::vertex_iterator]
    while {[$it has_next]} {
        set v [$it next]; set p [$v pin]; set name [get_full_name $p]
        set vals [list $name [$v level]]
        foreach rf {rise fall} {
            # Vertex.slew() reduces across all scenes, including when set_scene
            # is active. This interface explicitly selects the current scene.
            set text [$v slew_scenes_string $rf [list $scene_ptr] max false 12]
            lappend vals [expr {[sta::time_ui_sta $text]*1e12}]
        }
        foreach arrival [quad_pin_arrivals $v $scene] {
            lappend vals $arrival
            if {$arrival > -1e30 && $arrival < 1e30} {incr finite_arrivals}
        }
        set n [$p net]
        if {$n == "NULL"} {set term [$p term]; if {$term != "NULL"} {set n [$term net]}}
        if {$n == "NULL"} {lappend vals "" 0 0} else {
            lappend vals [get_full_name $n] [expr {[$n capacitance [sta::cmd_scene] max]*1e15}] [expr {[$n wire_capacitance [sta::cmd_scene] max]*1e15}]
        }
        puts $pf [join $vals "\t"]
        set ei [$v out_edge_iterator]
        while {[$ei has_next]} {
            set e [$ei next]
            set disabled 0
            foreach method {is_disabled_loop is_disabled_constraint is_disabled_constant is_disabled_cond_default is_disabled_bidirect_inst_path is_disabled_preset_clear} {
                if {[$e $method]} {set disabled 1}
            }
            set src [get_full_name [$e from_pin]]; set dst [get_full_name [$e to_pin]]
            foreach a [$e timing_arcs] {
                set fr [$a from_edge_name]; set to [$a to_edge_name]
                set elmore 0
                if {[$e role] == "wire"} {set elmore [expr {[sta::find_elmore [$e from_pin] [$e to_pin] $fr max]*1e12}]}
                puts $ef [join [list $src $dst [$e role] [$e sim_timing_sense] [$e cond] $fr $to [expr {[$e arc_delay $a [sta::cmd_scene] max]*1e12}] $disabled $elmore] "\t"]
            }
        }
        $ei finish
        incr pin_count
        if {$pin_count % 50000 == 0} {puts "QUAD_EXPORT_PROGRESS $scene $pin_count"; flush stdout}
    }
    $it finish
    close $pf; close $ef
    if {$finite_arrivals == 0} {error "Scene export has no finite arrival times"}
    puts "QUAD_EXPORT_ENDPOINTS $scene pins=$pin_count"; flush stdout
    quad_export_endpoints $directory $scene
    set f [open $directory/metrics.tsv w]
    puts $f "tns_ps\twns_ps"
    puts $f "[expr {[sta::total_negative_slack_scene_cmd $scene_ptr max]*1e12}]\t[expr {[sta::worst_slack_scene $scene_ptr max]*1e12}]"
    close $f
    puts "STAGE2_EXPORT_PASS $directory"
}
proc quad_export_endpoints {directory scene} {
    set f [open $directory/endpoints.tsv w]
    puts $f "pin\trf\trole\tat_ps\trequired_ps\tslack_ps\tmargin_ps\tcapture_pin\tcapture_rf\tcapture_at_ps"
    foreach rf {rise fall} {
        foreach end [find_timing_paths -scenes $scene -path_delay max -${rf}_to [sta::endpoints] -group_path_count 100000 -endpoint_path_count 1] {
            set actual_rf [$end end_transition]
            if {$actual_rf eq "^"} {set actual_rf rise}
            if {$actual_rf eq "v"} {set actual_rf fall}
            if {$actual_rf ni {rise fall}} {error "Unsupported endpoint transition: $actual_rf"}
            set role [$end check_role]
            # PathDelay endpoints can report role=setup without a Liberty setup
            # check (e.g. JPEG output ports). Their required-time boundary is
            # fixed because clock cells/locations remain protected.
            if {![$end is_check] && ([$end is_output_delay] || [$end is_path_delay])} {set role "output setup"}
            set capture_pin "";set capture_rf "";set capture_at 0
            set cp [$end target_clk_path]
            if {$cp ne "NULL"} {
                set capture_pin [get_full_name [$cp pin]]
                set capture_rf [$cp edge]
                if {$capture_rf eq "^"} {set capture_rf rise}
                if {$capture_rf eq "v"} {set capture_rf fall}
                if {$capture_rf ni {rise fall}} {error "Unsupported capture transition: $capture_rf"}
                set capture_at [expr {[$cp arrival]*1e12}]
            }
            puts $f [join [list [get_full_name [$end pin]] $actual_rf $role [expr {[$end data_arrival_time]*1e12}] [expr {[$end data_required_time]*1e12}] [expr {[$end slack]*1e12}] [expr {[$end margin]*1e12}] $capture_pin $capture_rf $capture_at] "\t"]
        }
    }
    close $f
}
