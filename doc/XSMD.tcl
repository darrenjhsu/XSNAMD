
# Want to force all atoms
set numatoms 1926
set num_q 76

# Load data
load ../../template/bin/1f6s_3/50kcal/XSMD.so XSMD

# Setup atom list ...
set atoms {}
for { set i 1 } { $i <= $numatoms } { incr i } {
    lappend atoms $i
}

# ...and tell NAMD the atoms we'll force
foreach atom $atoms {
    addatom $atom
}

set PI 3.1415926535898
set frame_num 0

# In this section these commands e.g. float_array are from XSMD.i which is the
# interface file in the /src/XSMD.i
set p_coord [float_array [expr {$numatoms * 3}]]
set p_force [float_array [expr {$numatoms * 3}]]
set p_force_old [double_array [expr {$numatoms * 3}]]
set p_scat [double_array [expr $num_q]]
set forces {}
set EMA_norm [double_array 1]
double_set $EMA_norm 0 0.0
# Initialize force - not necessary
set atm 1
for {set i 1} {$i <= $numatoms} {incr i} {
    float_set $p_force [expr {$i * 3 - 3}] 0.0
    float_set $p_force [expr {$i * 3 - 2}] 0.0 
    float_set $p_force [expr {$i * 3 - 1}] 0.0
    double_set $p_force_old [expr {$i * 3 - 3}] 0.0
    double_set $p_force_old [expr {$i * 3 - 2}] 0.0 
    double_set $p_force_old [expr {$i * 3 - 1}] 0.0
}
for {set i 0} {$i < $num_q} {incr i} {
    double_set $p_scat $i 0.0
}
if {[info exists XSMDrestartScat]} {
    # Read in stored scat 
    set XSMDrestartScatFile [open $XSMDrestartScat r]
    # Output the scattering S_old
    for {set i 0} {$i < $num_q} {incr i} {
        double_set $p_scat $i [gets $XSMDrestartScatFile]
    }
    close $XSMDrestartScatFile
    puts "Read from XSMD restart scat file..."
}
#if {[info exists XSMDrestartForce]} { 
#    set XSMDrestartForceFile [open $XSMDrestartForce r]
#    for {set i 0} {$i < [expr {$numatoms * 3}]} {incr i} {
#        double_set $p_force_old $i [gets $XSMDrestartForceFile]
#    }
#    close $XSMDrestartForceFile
#    puts "Read from XSMD restart force file..."
#}
if {[info exists XSMDrestartEMA]} {
    set XSMDrestartEMAFile [open $XSMDrestartEMA r]
    double_set $EMA_norm 0 [gets $XSMDrestartEMAFile]
    close $XSMDrestartEMAFile
    puts "Read from XSMD EMA file..."
}

proc calcforces { } {
    global atoms numatoms opt frame_num p_coord p_force p_force_old p_scat EMA_norm forces XSMDrestartFreq XSMDoutputName XSMDrestart force_this
    global PI num_q
    # Get coordinates and masses
    #loadmasses masses
    if {$opt == 1} {
        #puts "Frame is $frame_num"
        if {$frame_num % 50 == 0} {
            loadcoords coords
            for {set i 1} {$i <= $numatoms} {incr i} {
                float_set $p_coord [expr {$i * 3 - 3}] [lindex $coords($i) 0]
                float_set $p_coord [expr {$i * 3 - 2}] [lindex $coords($i) 1]
                float_set $p_coord [expr {$i * 3 - 1}] [lindex $coords($i) 2]
            }
        }
        XSMD_calc $p_coord $p_force $p_force_old $p_scat $frame_num $EMA_norm $XSMDrestart

        if {$frame_num % 50 == 0} {
            set forces {}
            set atm 1
            set numord [expr {$numatoms *3}]
            for {set i 0} {$i < $numord} {incr i} {
                lappend forces "[float_get $p_force $i] [float_get $p_force [incr i]] [float_get $p_force [incr i]]"
                incr atm
            }
        }
        foreach atom $atoms force $forces {	 
            addforce $atom $force
        }

        if {$frame_num % $XSMDrestartFreq == 0} {
           # Save restart files: scat curve and force
           set XSMDscat $XSMDoutputName.restart.XSMDscat
           #puts "File name is $XSMDscat"
           if {[file exists $XSMDscat]} {
               file rename -force -- $XSMDscat $XSMDscat.old
           }
           set XSMDscatFile [open $XSMDscat w]
           # Output the scattering S_old
           for {set i 0} {$i < $num_q} {incr i} {
               puts $XSMDscatFile [double_get $p_scat $i]
           }
           close $XSMDscatFile
           puts "Written to XSMD scat file..."
           #set XSMDforce $XSMDoutputName.restart.XSMDforce
           #if {[file exists $XSMDforce]} {
           #    file rename -force -- $XSMDforce $XSMDforce.old
           #}
           #set XSMDforceFile [open $XSMDforce w]
           #for {set i 0} {$i < [expr {$numatoms * 3}]} {incr i} {
           #    puts $XSMDforceFile [double_get $p_force_old $i]
           #}
           #close $XSMDforceFile
           #puts "Written to XSMD force file..."
           set XSMDEMA $XSMDoutputName.restart.XSMDEMA
           if {[file exists $XSMDEMA]} {
               file rename -force -- $XSMDEMA $XSMDEMA.old
           }
           set XSMDEMAFile [open $XSMDEMA w]
           puts $XSMDEMAFile [double_get $EMA_norm 0]
           close $XSMDEMAFile
           puts "Written to XSMD EMA file..."
        }
        incr frame_num 

        

    }
    return
}
