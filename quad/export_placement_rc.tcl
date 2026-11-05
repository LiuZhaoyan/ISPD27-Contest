# Geometry only: no placement, CTS, routing, or parasitic estimation.
proc placement_rc_export {path} {
    set block [ord::get_db_block]
    set scale [$block getDbUnitsPerMicron]
    set f [open $path w]
    puts $f "pin\tnet\tx_um\ty_um\tdirection\tlayer\tclock\tport"
    set it [sta::vertex_iterator]
    while {[$it has_next]} {
        set vertex [$it next]; set pin [$vertex pin]
        set net [$pin net]
        if {$net eq "NULL"} {
            set term [$pin term]
            if {$term ne "NULL"} {set net [$term net]}
        }
        if {$net eq "NULL"} {continue}
        set iterm [sta::sta_to_db_pin $pin]
        set dbnet [sta::sta_to_db_net $net]
        if {$dbnet eq "NULL"} {error "Missing physical net: [get_full_name $net]"}
        set layer 100; set port 0
        if {$iterm ne "NULL"} {
            lassign [$iterm getAvgXY] valid x y
            if {!$valid} {error "Missing pin geometry: [get_full_name $pin]"}
            set direction [$iterm getIoType]
            foreach mpin [[$iterm getMTerm] getMPins] {
                foreach box [$mpin getGeometry] {
                    set metal [$box getTechLayer]
                    if {$metal ne "NULL" && [$metal getRoutingLevel]>0} {
                        set layer [expr {min($layer,[$metal getRoutingLevel])}]
                    }
                }
            }
        } else {
            set bterms [$dbnet getBTerms]
            if {[llength $bterms]!=1} {error "Ambiguous physical port: [get_full_name $pin]"}
            set bterm [lindex $bterms 0]
            set port 1
            lassign [$bterm getFirstPinLocation] valid x y
            if {!$valid} {error "Missing port geometry: [get_full_name $pin]"}
            set direction [$bterm getIoType]
            foreach bpin [$bterm getBPins] {
                foreach box [$bpin getBoxes] {
                    set metal [$box getTechLayer]
                    if {$metal ne "NULL" && [$metal getRoutingLevel]>0} {
                        set layer [expr {min($layer,[$metal getRoutingLevel])}]
                    }
                }
            }
        }
        if {$layer==100} {error "Pin has no routing layer: [get_full_name $pin]"}
        puts $f [join [list [get_full_name $pin] [get_full_name $net] \
            [expr {double($x)/$scale}] [expr {double($y)/$scale}] \
            $direction $layer [expr {[$dbnet getSigType] eq "CLOCK"}] $port] "\t"]
    }
    $it finish
    close $f
}
