# Want to force all atoms
set numatoms 1749
set num_q 225 

# Load data
#source /home/djh992/MD_simulation/180607/XSMD/AllAtom/FF.tcl  
#source /home/djh992/MD_simulation/180607/XSMD/AllAtom/FF_E.tcl  
#source /home/djh992/MD_simulation/180607/XSMD/AllAtom/Ele_type.tcl  
#source /home/djh992/MD_simulation/180607/XSMD/AllAtom/dS1.tcl  
#source /home/djh992/MD_simulation/180607/XSMD/AllAtom/q.tcl  
#source /home/djh992/MD_simulation/180607/XSMD/AllAtom/tmp_scat_a_opt.tcl  
#source /home/djh992/MD_simulation/180607/XSMD/AllAtom/tmp_scat_b.tcl

load /home/djh992/MD_simulation/180607/CUDA_FUN_float_cytc/src/XSMD.so XSMD

# Setup atom list ...
set atoms {}
for { set i 1 } { $i <= $numatoms } { incr i } {
    lappend atoms $i
}

# ...and tell NAMD the atoms we'll force
foreach atom $atoms {
    addatom $atom
}


# Take acceleration factors from NAMD config file, convert to kcal/(mol*Ang*amu)
set linaccel_namd [vecscale [expr 1.0/418.68] $linaccel]
set angaccel_namd [expr double($angaccel)/418.68]
set PI 3.1415926535898
set frame_num 0
set p_coord [float_array [expr {$numatoms * 3}]]
set p_force [float_array [expr {$numatoms * 3}]]
set p_scat [double_array [expr $num_q]]
set EMA_norm [double_array 1]
double_set $EMA_norm 0.0


proc calcforces { } {
    global atoms numatoms opt frame_num
    #linaccel_namd angaccel_namd qs Sb Sa dS Ele FF_E
    global PI
#    global bead_coord basis
    # Get coordinates and masses
    loadcoords coords
    loadmasses masses
    if {$opt == 1} {
        set atm 1
        for {set i 1} {$i <= $numatoms} {incr i} {
            float_set $p_coord [expr {$i * 3 - 3}] [lindex $coords($i) 0]
            float_set $p_coord [expr {$i * 3 - 2}] [lindex $coords($i) 1]
            float_set $p_coord [expr {$i * 3 - 1}] [lindex $coords($i) 2]
        }
    #    puts [expr float([lindex $coords(1) 0])]
        #puts $p_force 
        # Need to turn these into pointers 
        # void XSMD (float *coord, float *Force)
        XSMD_calc $p_coord $p_force $p_scat $frame_num $EMA_norm
        incr frame_num 
        # unpack force
        #set force {}
        #for {set i 0} {$i < [expr $numatoms * 3]} {incr i} {
            #puts [float_get $p_force $i]
            #puts $i
        #    set tmp [float_get $p_force $i]
            #puts $tmp
        #    lappend $force $tmp
        #}
    
        set atm 1
        for {set i 0} {$i < [expr {$numatoms * 3}]} {set i [expr {$i + 3}]} {
            set f_i {}
            lappend f_i [float_get $p_force $i]
            lappend f_i [float_get $p_force [expr {$i + 1}]]
            lappend f_i [float_get $p_force [expr {$i + 2}]]
           #addforce $atm [vecscale [expr 1/$masses($atm)] $f_i]
           #addforce $atm [vecscale $masses($atm) $f_i]
            addforce $atm $f_i
            incr atm
         
        }

    }
    return
}
