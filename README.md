
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
currently set as adjustable just as in FoXS. 
<!-- not uniform. It is with HyPred approach (radial sum of electron density
difference up to vdW radii + 3 A). -->

The atomic form factors in vacuum are calculated using Waasmaier-Kirfel table.

The volume for dummy atom calcualtion were taken from Svergun 1995 J Appl 
Crystallgr paper, which refers to Fraser 1978 J Appl Crystallgr paper and 
International Tables for X-ray Crystallography (1968). 

In the surface area calculation part the spiral generating function is from 
Bauer 1998 Guidance Navigation and Control Conference and Exhibit paper. 

The surface area calculation is done numerically following J Appl Crystallgr 
1983 Connolly "Analytical Molecular Surface Calculation." Rasterized points 
sample the vdW sphere, which has to be outside of any other vdW spheres of 
other atoms. Extended by solvent radius, the point (solvent center) must also 
be far enough from the vdW spheres of other atoms.

## Why use this program?

1. It takes care of changing scattering intensity due to surface area
   variation. This is necessary if you have an (un)folding process to explore.
   For example, a study focussing on molten globule state will benefit a lot.

1. It is fast. If force is evaluated every 50 steps, the computational
   overhead is almost negligible!

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

### If you have an experimental measurement

1. An input python file **input.py** contains pointers to the PDB and PSF 
   files (that you probably generated through `<psfgen`> in VMD), scattering
   profiles (q, S\_q, dS\_q and S\_err if you have it), and a few parameters. 
   This is the file to edit when changing the parameters. Please refer to the
   comments and instructions in the file.

1. Run `make cu` to parse the PDB and PSF files and generate
   `mol_param.cu` and `coord_ref.cu`. 

1. Run `make fit` to compile `fit_initial.cu` 

1. Submit your `fit_initial.sh` to calculate c1 and c2 that best fit your
   data. This will also generate the `scat_param.cu` with the data you
   provided properly scaled to the calculated curve. 

1. Run `make` to finally compile the `XSMD.cu` to a shared object that NAMD
   will call every step. 

### If you have two structures and you want to drive one to another

1. Modify the **input.py** for appropriately.

1. Run `make cu` to parse the PDB and PSF files and generate
   `mol_param.cu` and `coord_ref.cu`. 

1. Run `make initial` to compile `structure_calc.cu`

1. Submit your `structure_calc.sh` to calculate the difference signal.

It is also possible to use an equilibrium trajectory or an ensemble of
structures to calculate the target scattering profile. In this case, run
`make traj` and submit `traj_scatter.sh` to get the curve and difference. 

1. Run `make` to finally compile the `XSMD.cu` to a shared object that NAMD
   will call every step. 
 



Of the files, 

Init\_calc.cu       calculates with given two sets of coordinates, the scattering pattern for reference (S\_ref) and for initial (S\_init) structures, and then compute the difference (dS = S\_ref - S\_init)

raster.cu          specifies parameters. 
kernel.cu          is the workhorse of the package.
XSMD.cu            is where the code NAMD calls every step is in.
speedtest.cu       is to test new features. 

