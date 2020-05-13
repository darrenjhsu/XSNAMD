
## Example datasets and tutorial

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

#### Getting the XSMD.so

You need to load the nvcc compiler. On our local cluster we used `module load
cuda`. Yours may differ.

1. Edit the `make_input`: `data_path=data/Ala10`. Save the file.

1. Do `./make_input` to copy all .cu files to src/ and .hh files to include/.

1. Build the file with `make DSET=Ala10 KCHI=1e-5`. This file will be place in
   `bin/Ala/1e-5/XSMD.so` along with the backup code. You need the swig
   interface to be able to compile the wrap file.

1. Edit the XSMD.tcl so that the XSMD path is correctly put in.

1. Run simulation. You should see the chi square output in the log file.

