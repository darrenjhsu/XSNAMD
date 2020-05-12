
### XSMD module - add force based on XSP

# Want to force all atoms; look at your molecule and change accodingly
set numatoms 104

# Number of q points; look at your scattering profile and change accordingly
set num_q 301

# Load the code for calculating scattering patterns and force.s
# Change to wherever the XSMD.so is.
load ../bin/Ala10/1e-5/XSMD.so XSMD

# Setup atom list
set atoms {}
for { set i 1 } { $i <= $numatoms } { incr i } {
    lappend atoms $i
}

# ... and tell NAMD the atoms we'll exert force
foreach atom $atoms {
    addatom $atom
}

# Set some parameters
set PI 3.1415926535898
set frame_num 0
set p_coord [float_array [expr {$numatoms * 3}]]
set p_force [float_array [expr {$numatoms * 3}]]
set p_force_old [double_array [expr {$numatoms * 3}]]
set p_scat [double_array [expr $num_q]]
set forces {}

# Set exponential moving averaging
set EMA_norm [double_array 1]
double_set $EMA_norm 0 0.0
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

# If there are restart files, read in
if {[info exists XSMDrestartScat]} {
    # Read in stored scat 
    set XSMDrestartScatFile [open $XSMDrestartScat r]
    # Output the scattering S_old
    for {set i 0} {$i < $num_q} {incr i} {
        double_set $p_scat $i [gets $XSMDrestartScatFile]
    }
    close $XSMDrestartScatFile
    puts "Finished reading from XSMD restart scat file..."
}

# If restarting from a EMA run, then also include that coeff.
if {[info exists XSMDrestartEMA]} {
    set XSMDrestartEMAFile [open $XSMDrestartEMA r]
    double_set $EMA_norm 0 [gets $XSMDrestartEMAFile]
    close $XSMDrestartEMAFile
    puts "Read from XSMD EMA file..."
}

proc calcforces { } {
    global atoms numatoms opt frame_num p_coord p_force p_force_old p_scat EMA_norm forces XSMDrestartFreq XSMDoutputName XSMDrestart
    global PI num_q
    # You could delay the start of XSMD by setting opt to 0 for the
    # first part of simulations.
    if {$opt == 1} {
        if {$frame_num % 50 == 0} {
            # If you have delta_t less than 50, change the 50 above accordingly
            # Get coordinates and masses
            loadcoords coords
            for {set i 1} {$i <= $numatoms} {incr i} {
                float_set $p_coord [expr {$i * 3 - 3}] [lindex $coords($i) 0]
                float_set $p_coord [expr {$i * 3 - 2}] [lindex $coords($i) 1]
                float_set $p_coord [expr {$i * 3 - 1}] [lindex $coords($i) 2]
            }
        }

        # Passing all info to the C code and calculate scattering pattern
        # and force. Note if you don't have GPU this will return 0
        # and nothing will happen.
        XSMD_calc $p_coord $p_force $p_force_old $p_scat $frame_num $EMA_norm $XSMDrestart

        if {$frame_num % 50 == 0} {
            set forces {}
            set atm 1
            set numord [expr {$numatoms *3}]
            for {set i 0} {$i < $numord} {incr i} {
                # Append force
                lappend forces "[float_get $p_force $i] [float_get $p_force [incr i]] [float_get $p_force [incr i]]"
                incr atm
            }
        }
        # Tell NAMD to add force
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
