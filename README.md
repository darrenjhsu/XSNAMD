
## Driving an MD structure to another with SAXS signal.

Contact: Darren Hsu (darrenhsu2015 at u.northwestern.edu)

### Problem statment 
In time-resolved X-ray solution scattering, researchers
obtain difference signal as a function of system evolution. In my case, a
protein assumes different states and gives distinct difference scattering
signals. We want to find some structures that give the difference signal that
match the difference signal.

### What's being done 
We input the X-ray scattering signal as a
constraint in the MD simulation. The program calculates X-ray scattering
signals at each defined snapshot, compare it to the reference signal and calculate the
difference signal. It then compare the calculated difference signal to the input one, and
it uses the deviation between them to derive force that acts on each atom. The math is
described in [our paper](https://doi.org/10.1063/5.0007158). 


## Why use this program?

1. It takes care of scattering intensity changes due to surface area
   variation. This is necessary if you have an (un)folding process to explore.
   For example, a study focusing on molten globule states will benefit a lot.

1. It is fast. If force is evaluated every 50 steps, the computational
   overhead is about 3 - 4 %.

1. Only one simulation box is required. You solvate the protein, equlibrate it,
   fit to the static scattering data, and then fit to the difference data.

1. It is compatible with some enhanced sampling methods such as metadynamics.
   This again comes in handy when exploring the conformational space for an
   unfolding reaction, while the correction force keeps the structure from
   deviating from the difference signal. 

## Requirements

### Hardware

This program is written in CUDA C and runs only on Nvidia GPUs. It assumes
that you have access to an Nvidia GPU. At the time of writing, this code has
been tested on K40, K80, P100, and Quadro 6000 cards.

### Software dependencies

1. For some parts **GNU Scientific Library (GSL) >= 2.5** is required. Your cluster
may have it installed. In that case, set the path in the Makefile to it.
Otherwise, you need to install a local copy of GSL and point to the directory
in the Makefile. More on that in the Installation section. 

1. Nvidia's **nvcc** compiler is required. Consult your system administrator about
cuda availability on your machine. At the time of writing we are using cuda/9.1.85. 

1. **SWIG >= 3.0.x** interface is required. It's likely already installed on your machine. 
See [SWIG
website](http://www.swig.org/Doc3.0/Preface.html#Preface_unix_installation) for more
inforamtion about installing it locally on your server.

1. There is a basic python (**python 3, numpy, scipy**) files for converting pdb to 
input C code. You should not need an env for running the provided python script (input.py).

Since the code is intended to be a module of NAMD, **NAMD (CUDA build)** should be 
installed on your machine. This code is tested with NAMD 2.11.
You need to know the path to executable `namd2`.

**Note about NAMD 2.12 and 2.13**: It seems that the pre-built binary version
of NAMD 2.12 and 2.13 (both multicore-CUDA and ibverbs builds) have a different
behavior when its TCL script loads `XSMD.so` which generates errors regarding
undefined symbols such as `Tcl_GetStringFromObj`. To run on those versions you need
to compile the NAMD from source code using the TCL library they provide.
Basically follow the instruction in their release note.

## Installation

Simply download this repo by `git clone https://github.com/darrenjhsu/XSNAMD`
on your machine. It creates a directory called XSNAMD/ in your working directory.

Now follow the READMEs in both root directory and example/ for more information.

### Example datasets / Tutorial

Example datasets and a tutorial can be found in the example/ folder. The input
files are prepared for you, so you can try out the program.

## Basic application workflow

1. Prepare a run-able NAMD simulation system (see [NAMD tutorials](http://www.ks.uiuc.edu/Training/Tutorials/namd-index.html)
for more information if you are not familiar with NAMD simulations)

1. After that, prepare (see below) experimental data so that you can compile the `XSMD.so` file.

1. Re-run the NAMD simulation with tclForce turned on and points to the `XSMD.so`
while running simulations.


### File structures

```
XSNAMD/
|   input.py
|   make_input
|   Makefile
|   Readme.md
|---bin/
|   |---PROT1/
|           speedtest.out
|           fit_initial.out
|           fit_traj_initial.out
|           1e-5/            # Each k_chi will have its own directory. Might change.
|               XSMD.so
|               backup_code/
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


## Application scenarios

### If you have two known structures, and you want to test if the program can drive one structure to another

You will calculate the scattering signals from both structures and prepare a mock dataset of q, S\_ref, and dS.

Good for test runs and for understanding how the program works. Refer to the README in example/ for more information.


<!-- ### If you want to fit c, c1, and c2 with one starting strcuture and one static experimental measurement

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
-->


### If you have one known structure, a static measurement, and a difference signal to fit

It is encouraged that you at least do an equilibrium run for the structure and fit the
average of that run to the static measurement for c, c1 and c2. To do that see next section.

1. Assuming the input data are prepared in data/PROT1/, where PROT1 is your 
   protein's name.

1. An input python file **input.py** contains pointers to the PDB and PSF 
   files (that you probably generated through `psfgen` in VMD), absolute scattering
   profiles (q, S(q), Serr(q)), difference scattering profiles (q, dS(q), and dSerr(q)) 
   if you have it, and a few parameters. 
   This is the file to edit when changing the parameters. Please refer to the
   comments and instructions in the file.

1. Run `python input.py` to parse the PDB and PSF files and generate
   `mol_param.cu/hh` and `coord_ref.cu/hh`

1. Run `make fit DSET=PROT1` to compile `fit_initial.cu`. In doing so an .out
   file called `fit_initial.out` will be placed in `bin/PROT1/`. 
   The Makefile reads the `src` and also `data/PROT1` for all relevant source
   code files.

1. Execute `fit_initial.out` to calculate c1 and c2 that best fit your data.
   See the **Input arguments** section below for input arguments for this .out file.
   On an HPC this is usually done through a job submission script. Yours may
   vary. This will also generate the`scat_param.cu/hh` with the data you provided 
   properly scaled to the calculated curve, in the directory `data/PROT1`.

1. Run `make KCHI=1e-5 DSET=PROT1` to finally compile the `XSMD.cu` 
   to a shared object that NAMD will call every step. 
   DSET is the name of your dataset.
   KCHI can be just a string. It does not need to be the actual spring
   constant. It simply determines where in the `bin` the `XSMD.so` will be
   placed. The actual spring constant is dictate in the
   `data/PROT1/env_param.cu`. For example, KCHI can be `20kcal` so you know you
   are supplying 20 kcal/mol of initial X-ray scattering related potential.

1. When running NAMD, add these keywords in your NAMD config files. See the
   examples for more information.
```
    tclForces            on
    set opt              0             # 1 when you are ready to turn it on
    tclforcesscript      XSMD.tcl      # See /doc/XSMD.tcl for details
    set XSMDrestartFreq  5000          # Will write to .restart.XSMDscat and restart.XSMDEMA files 
    set XSMDoutputName   $outputname
    set XSMDrestart      0             # 1 if it's a continuing XSMD run
    if {$XSMDrestart} {
        set XSMDrestartScat  $outputname.restart.XSMDscat
        set XSMDrestartEMA   $outputname.restart.XSMDEMA
    }
```

During the simulation, look at the log file. It should show the chi square decreasing gradually. 


### If you have an equilibrium run of one protein, a static measurement, and a difference signal to fit

This is the ideal setting and is the typical case of a real application.

1. Assuming the input data are prepared in data/PROT1/, where PROT1 is your 
   protein's name.

1. An input python file **input.py** contains pointers to the PDB and PSF 
   files (that you probably generated through `psfgen` in VMD), scattering
   profiles (q, S\_q, dS\_q and S\_err if you have it), and a few parameters. 
   This is the file to edit when changing the parameters. Please refer to the
   comments and instructions in the file.

1. Run `python input.py` to parse the PDB and PSF files and generate
   `mol_param.cu/hh` and `coord_ref.cu/hh`. 

1. Export the atomic coordinates of the solute (a.k.a. the atoms you're using
   to calculate the scattering pattern) from the equilibrium run as a
   multiframe xyz file named `trajectory.xyz` to the root folder.

1. Run `make fit_traj DSET=PROT1` to compile `fit_traj_initial.cu`. This will
   compile files in `data/PROT1/` and `src` to produce
   `bin/PROT1/fit_traj_initial.out`.

1. Execute `fit_traj_initial.out` to calculate c1 and c2 that best fit your
   data using the trajectory as the ground state ensemble.
   See the **Input arguments** section below for input arguments for this .out file.
   On an HPC this is usually done through a job submission script. Yours may
   vary. This will also generate the`scat_param.cu/hh` with the data you provided 
   properly scaled to the calculated curve, in the directory `data/PROT1`.

1. Run `make KCHI=1e-5 DSET=PROT1` to finally compile the `XSMD.cu` 
   to a shared object that NAMD will call every step. 
   DSET is the name of your dataset.
   KCHI can be just a string. It does not need to be the actual spring
   constant. It simply determines where in the `bin` the `XSMD.so` will be
   placed. The actual spring constant is dictate in the
   `data/PROT1/env_param.cu`. For example, KCHI can be `20kcal` so you know you
   are supplying 20 kcal/mol of initial X-ray scattering related potential.

1. When running NAMD, add these keywords in your NAMD config files.
```
    tclForces            on
    set opt              0             # 1 when you are ready to turn it on
    tclforcesscript      XSMD.tcl      # See /doc/XSMD.tcl for details
    set XSMDrestartFreq  5000          # Will write to .XSMDscat and .XSMDEMA files 
    set XSMDoutputName   $outputname
    set XSMDrestart      0             # 1 if it's a continuing XSMD run
    if {$XSMDrestart} {
        set XSMDrestartScat  $outputname.restart.XSMDscat
        set XSMDrestartEMA   $outputname.restart.XSMDEMA
    }
```

During the simulation, look at the log file. It should show the chi square decreasing gradually. 

### Combining the XSMD simulations with metadynamics

Manuals coming ...



## Input arguments for different .out files

-  `fit_initial.out` takes 6 arguments (c1 initial, c1 step, c1 end, c2
   initial, c2 step, and c2 end) to define a scan range. For example:
   
   `./fit_initial.out 0.95 0.01 1.05 0.0 0.1 4.0`

-  `fit_initial.out` takes 9 arguments (trajectory file name, nubmer of frames
   in that file, how many frames (from the last) to use for fitting, c1 initial, 
   c1 step, c1 end, c2 initial, c2 step, and c2 end) to define a scan range. For example:
   
   `./fit_traj_initial.out mytraj.xyz 2000 500 0.95 0.01 1.05 0.0 0.1 4.0`

   You can cheat this program to use a middle section by setting the number of
   frames smaller than actual number of frames. For example, you have a 2000
   frame trajectory `mytraj.xyz`, you can use the frames 1000 - 1500 by:

   `./fit_traj_initial.out mytraj.xyz 1500 500 0.95 0.01 1.05 0.0 0.1 4.0`

-  `XSMD.so` takes 7 arguments while being called by the XSMD.tcl. They are: 
   (1) an array of coordinates, (2) an array of force, (3) an array of old force (not used),
   (4) and array of scattering intensity, (5) frame number, (6) normalizing
   constant of exponential moving averaging, and (7) whether XSMD is a restart
   run. Ideally you don't need to worry about these arguments.

## Important source files and where to find them

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
are used in the `FF_calc` kernel in `kernel.cu`. This file is different for
every system you want to run simulations on.

### `env_param.cu`

Tells some environmental parameters such as k the spring constant (that is 
inherited all the way from the initial `input.py`) or rho the solvent electron
density (0.334 for water at 20 deg C). delta\_t and tau are defined in the
paper. This file is different for every system you want to run simulations on.

### `scat_param.cu`

Contains number of q points, c1, c2, c, scattering patterns of the
reference structure (or ensemble run), the difference pattern scaled to the
scattering magnitude of reference signal, and error of the difference signal
also scaled. Note that c is not used in the actual calculation. This file is
different for every system you want to run simulations on.

## Under the hood

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

