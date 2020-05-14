
# Example datasets and tutorial

## Choose a system you want to have fun with

In this folder a portion of the work published in JCP is converted to example
datasets. The two examples are (1) Ala10 folding and (2) LAO transition. Please
refer to the JCP paper for more information.

### Ala10 folding 

Ala10 has two conformations, a random coil and an alpha helix. Our goal here is
to drive the random coil to the alpha helix, given the X-ray solution
scattering signal generated from the random coil to alpha helix transition.

Input files for XSMD simulation are provided in the Ala10/ folder.
If you turn off the `tclforces` in the config file, it will run as a normal MD
simulation. If you enable the `tclforces`, then you enable the XSMD.

To prepare the `XSMD.so` object that is referred in the `XSMD.tcl`, you need to
compile the code in src/ folder.

### LAO structural transition

LAO also has two conformations, one is the holo state (PDB: 1lst), and the
other the apo state (PDB: 2lao). In this simulation we're driving 1lst (without
its substrate) to 2lao. 


## Compiling the XSMD.so

You need to load the nvcc compiler. On our local cluster we used `module load
cuda`. Yours may differ.

1. Edit the `make_input`: `data_path=data/Ala10`. Save the file.

1. Do `./make_input` to copy all .cu files to src/ and .hh files to include/.

1. Build the file with `make DSET=Ala10 KCHI=1e-5`. The file will be placed in
   `bin/Ala/1e-5/XSMD.so` along with the backup code. You need the swig
   interface to be able to compile the wrap file. If you don't have swig
   installed globally, you can install it locally and include that to your
   path. (`export PATH=$PATH:/path/to/swig`)


## Run a simulation with XSMD.so attached to it

1. Edit the XSMD.tcl by `vim XSMD.tcl` so that the XSMD path is correctly put in.
   The line you need to change is under `# Change to wherever the XSMD.so is.`
   The result looks like `load path/to/your/XSMD.so XSMD`

1. Edit the {YOURSYSTEM}\_XSMD\_0.conf. Make sure these parameters are correct:
   ```
   tclforces           on
   set opt             0
   tclforcesscript     XSMD.tcl    ;# Relative path to where this config file is
   ```
   as well as 
   ```
   set XSMDrestartFreq  5000
   set XSMDoutputName   $outputname
   set XSMDrestart      0   ;# new XSMD simulation
   set XSMDrestartScat  $outputname.restart.XSMDscat
   set XSMDrestartEMA   $outputname.restart.XSMDEMA
   ```

1. Run simulation. The method to run NAMD simulations differs a lot across
   clusters, so please consult your system administrator about this part.
   In short, if you have globally installed binary or a module, you may be able
   to simply issue `namd2` as your command, followed by options, and then by
   input file name. For clusters with a job queueing system, it's common to
   prepare a submission script in which you issue `namd2` to run simulations.

1. Inspect the log file ({YOURSYSTEM}\_XSMD\_0.log) 
   You should see the chi square output in the log file. The chi square should
   decrease dramatically when the force starts to ramp up. The convex contact area 
   should drop from 185 A^2 to 147 A^2 for the Ala10 system. 
   You can use other python packages to analyze the trajectory. 
