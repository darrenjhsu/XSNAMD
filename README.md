
## Driving an MD structure to another with SAXS signal.

This project is to use GPU to accelerate X-ray scattering calculation with
Debye formula looping over atoms and use it in MD simulation. Concepts were 
taken from Bjorling et al. JCTC 2015, 11, 780 and exponential moving average is
taken from Chen & Hub, Biophysics J, 2015. 

In short, at each interval we evaluate the scattering profile, and take the
negative gradient (definition of force) of chi square, which is a function of
all coordinates (only). 

The calculation of scattering profile is the same as in FoXS but the form 
factors are calculated explicitly. The solvation shell contrast coefficient is
currently set as adjustable just as in FoXS: Schneidman-Duhovny et al, NAR 2010.
<!-- not uniform. It is with HyPred approach (radial sum of electron density
difference up to vdW radii + 3 A). -->

The atomic form factors in vacuum are calculated using Waasmaier-Kirfel table.

The volume for dummy atom calcualtion were taken from Svergun 1995 J Appl 
Crystallgr paper, which refers to Fraser 1978 J Appl Crystallgr paper and 
International Tables for X-ray Crystallography (1968). 


The surface area calculation is done numerically following J Appl Crystallgr 
1983 Connolly "Analytical Molecular Surface Calculation." Rasterized points 
sample the vdW sphere, which has to be outside of any other vdW spheres of 
other atoms. Extended by solvent radius, the point (solvent center) must also 
be far enough from the vdW spheres of other atoms.

In the surface area calculation part the spiral generating function is from 
Bauer 1998 Guidance Navigation and Control Conference and Exhibit paper. 

## Why use this program?

1. It takes care of changing scattering intensity due to surface area
   variation. This is necessary if you have an (un)folding process to explore.
   For example, a study focussing on molten globule state will benefit a lot.

1. It is fast. If force is evaluated every 50 steps, the computational
   overhead is almost negligible.

1. Only one simulation box is required. You solvate the protein, equlibrate it,
   fit to the static scattering data, and then fit the difference data.

1. It is compatible with some enhanced sampling methods such as metadynamics.
   This again comes in handy when exploring the conformational space for an
   unfolding reaction, while the correction force keeps the structure from
   deviating from the difference signal. 

## Requirements

This program is written in CUDA C and runs only on Nvidia GPUs.

For some parts GNU Scientific Library (GSL) is required. 

There are some python files too. 


## Application workflow

There are several steps. This is given that you work on a cluster with Nvidia
GPU access, and that you use some sort of job submission system.

### File structures

```
Root Folder
|   input.py
|   make_input
|   Makefile
|   Readme.md
|---bin/
|   |---PROT1/
|           speedtest.out
|           fit_initial.out
|           fit_traj_initial.out
|           1e-5/
|               XSMD.so
|   |---PROT2/
|           # Similar structures as PROT1/
|---data/
|   |---PROT1/
|           PROT1.psf
|           PROT1.pdb
|           S_exp.txt
|           dS_exp.txt
|           mol_param.cu/hh   # Will be generated along prep
|           scat_param.cu/hh  # Will be generated along prep
|           env_param.cu/hh   # Will be generated along prep
|           coord_ref.cu/hh   # Will be generated along prep
|   |---PROT2/
|           # Similar structures as PROT2
|---include/
|       *.hh files
|---lib/
|---src/
|       *.cu files

```


### If you want to fit c, c1, and c2 with one starting strcuture and one static experimental measurement

1. Assuming the input data are prepared in data/PROT1/, where PROT1 is your 
   protein's name.

1. An input python file **input.py** contains pointers to the PDB and PSF 
   files (that you probably generated through `<psfgen`> in VMD), scattering
   profiles (q, S\_q, dS\_q and S\_err if you have it), and a few parameters. 
   This is the file to edit when changing the parameters. Please refer to the
   comments and instructions in the file.

1. Run `python input.py` to parse the PDB and PSF files and generate
   `mol_param.cu/hh` and `coord_ref.cu/hh`. 

1. Edit the `data_path` variable to PROT1 in `make_input` which is a bash 
   script file, and execute it by `./make_input` to copy the .cu/hh files from
   /data/PROT1 to /src .    

1. Run `make fit DSET=PROT1` to compile `fit_initial.cu` 

1. Submit your `fit_initial.sh` that executes `fit_initial.out` to calculate 
   c1 and c2 that best fit your data. This will also generate the 
   `scat_param.cu/hh` with the data you provided properly scaled to the 
   calculated curve. 

1. Run `make KCHI=1e-5 DSET=PROT1` to finally compile the `XSMD.cu` 
   to a shared object that NAMD will call every step. 

   KCHI is the spring constant that you determine the magnitude of the X-ray
   scattering potential, and DSET is the name of your dataset.


### If you have two known structures, and you want to test if the program can drive one structure to another

Good for test runs. Manuals coming ...

### If you have one known structure, a static measurement, and a difference signal to fit

It is encouraged that you do an equilibrium run for the structure and fit the
average of that run to the static measurement. To do that see next section.

1. Assuming the input data are prepared in data/PROT1/, where PROT1 is your 
   protein's name.

1. An input python file **input.py** contains pointers to the PDB and PSF 
   files (that you probably generated through `<psfgen`> in VMD), scattering
   profiles (q, S\_q, dS\_q and S\_err if you have it), and a few parameters. 
   This is the file to edit when changing the parameters. Please refer to the
   comments and instructions in the file.

1. Run `python input.py` to parse the PDB and PSF files and generate
   `mol_param.cu/hh` and `coord_ref.cu/hh`. 

1. Edit the `data_path` variable to PROT1 in `make_input` which is a bash 
   script file, and execute it by `./make_input` to copy the .cu/hh files from
   /data/PROT1 to /src .    

1. Run `make fit DSET=PROT1` to compile `fit_initial.cu`

1. Submit your `fit_initial.sh` that executes `fit_initial.out` to calculate 
   c1 and c2 that best fit your data. This will also generate the 
   `scat_param.cu/hh` with the data you provided properly scaled to the 
   calculated curve. 

1. Run `make KCHI=1e-5 DSET=PROT1` to finally compile the `XSMD.cu` 
   to a shared object that NAMD will call every step. 

   KCHI is the spring constant that you determine the magnitude of the X-ray
   scattering potential, and DSET is the name of your dataset.

1. When running NAMD, add these keywords in your NAMD config files.
```
    tclForces        on
    set opt             0             # 1 when you are ready to turn it on
    tclforcesscript     XSMD.tcl      # See /doc/XSMD.tcl for details
    set XSMDrestartFreq 5000          # Will write to .XSMDscat and .XSMDEMA files 
    set XSMDoutputName  $outputname
    set XSMDrestart     0             # 1 if it's a continuing XSMD run
```

1. During the simulation, look at the log file. It should show the chi square
   decreasing gradually. 


### If you have an equilibrium run of one protein, a static measurement, and a difference signal to fit

This is the ideal setting.

1. Assuming the input data are prepared in data/PROT1/, where PROT1 is your 
   protein's name.

1. An input python file **input.py** contains pointers to the PDB and PSF 
   files (that you probably generated through `<psfgen`> in VMD), scattering
   profiles (q, S\_q, dS\_q and S\_err if you have it), and a few parameters. 
   This is the file to edit when changing the parameters. Please refer to the
   comments and instructions in the file.

1. Run `python input.py` to parse the PDB and PSF files and generate
   `mol_param.cu/hh` and `coord_ref.cu/hh`. 

1. Export the atomic coordinates of the solute (a.k.a. the atoms you're using
   to calculate the scattering pattern) from the equilibrium run as a
   multiframe xyz file named `trajectory.xyz` to the root folder.

1. Edit the `data_path` variable to PROT1 in `make_input` which is a bash 
   script file, and execute it by `./make_input` to copy the .cu/hh files from
   /data/PROT1 to /src .    

1. Run `make fit_traj DSET=PROT1` to compile `fit_traj_initial.cu`

1. Submit your `fit_traj_initial.sh` that executes `fit_traj_initial.out` to 
   calculate c, c1 and c2 that best fit your data. This will also generate the 
   `scat_param.cu/hh` with the data you provided properly scaled to the 
   calculated curve.

1. Run `make KCHI=1e-5 DSET=PROT1` to finally compile the `XSMD.cu` 
   to a shared object that NAMD will call every step. 

   KCHI is the spring constant that you determine the magnitude of the X-ray
   scattering potential, and DSET is the name of your dataset.

1. When running NAMD, add these keywords in your NAMD config files.
```
    tclForces        on
    set opt             0             # 1 when you are ready to turn it on
    tclforcesscript     XSMD.tcl      # See /doc/XSMD.tcl for details
    set XSMDrestartFreq 5000          # Will write to .XSMDscat and .XSMDEMA files 
    set XSMDoutputName  $outputname
    set XSMDrestart     0             # 1 if it's a continuing XSMD run
```

1. During the simulation, look at the log file. It should show the chi square
   decreasing gradually. 

### Combining the XSMD simulations with metadynamics

Manuals coming ...


## Important source files

### `XSMD.cu`

Contains a function which will be compiled into a .so shared object file. In
this function kernels will be sequentially called to calculate solute surface
area, scattering patterns, and forces that should be applied to the solute
atoms.

### `kernel.cu`

Contains kernel functions that actually perform scattering calculations.

### `XSMD.i`

The interface file that goes along with SWIG, which enables tcl to call a C
function. It also provides some helper functions such as declaring float arrays
in C from tcl side.

### `speedtest.cu`

A file that looks like `XSMD.cu` but reads in coordinates from `coord.cu`. This
allows you to test the functionality of these kernels (as well as do speed
tests if you use nvprof in the HPC job submission.)

### `mol_param.cu`

Tells number of atoms and atom types. Atom types are defined in WaasKirf.cu and
are used in the `FF_calc` kernel in `kernel.cu` 

### `env_param.cu`

Tells some environmental parameters such as k the spring constant (that is 
inherited all the way from the initial `input.py`) or rho the solvent electron
density (0.334 for water at 20 deg C). delta\_t and tau are defined in the
paper.

### `scat_param.cu`

Contains number of q points, c1, c2, c, scattering patterns of the
reference structure (or ensemble run), the difference pattern scaled to the
scattering magnitude of reference signal, and error of the difference signal
also scaled. Note that c is not used in the actual calculation. 


<!--Of the files, 

Init\_calc.cu       calculates with given two sets of coordinates, the scattering pattern for reference (S\_ref) and for initial (S\_init) structures, and then compute the difference (dS = S\_ref - S\_init)

raster.cu          specifies parameters. 
kernel.cu          is the workhorse of the package.
XSMD.cu            is where the code NAMD calls every step is in.
speedtest.cu       is to test new features. 
-->

