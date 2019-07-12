
import numpy as np
import os.path
import re
import math
import platform
import scipy.signal as sig
from scipy import interpolate

print(platform.python_version())

def next_2048(x):
    #return 1 if x == 0 else int(2**math.ceil(math.log(x,2)))
    print(((x+2047)/2048)*2048)
    return (((x+2047)/2048)*2048)



## Specify parameters

# Driving modes. Driving the initial structure to
# 'c': another crystal strcuture, 't': an average of trajectory,
# 's': static SAXS signal, 'd': difference SAXS signal
driving_mode = 's'

# Initial PDB and PSF files
data_path = 'data/1f6s_3/'
fpsf = data_path + '1f6s_autopsf.psf'
fpdb = data_path + '1f6s_autopsf.pdb'

#data_path = 'data/1oad/'
#fpsf = data_path + '1oad_autopsf.psf'
#fpdb = data_path + '1oad_autopsf.pdb'

# Number of atoms
num_atom = 1926

## Experimental files (file format: q, S_exp [, S_err])
# One static (S_exp) is required, and many difference can follow.
S_exp_file = data_path + 'S_exp60.txt'
dS_exp_file = data_path + 'dS_exp60_full_SADS.txt'

# Unit of the q vector. If your data is from SASBDB, it is probably 1 / nm.
q_unit = 'A' # Options: 'nm' or 'A'

num_dS = 1   # Number of dS components. Has no effect when driving mode is not set to 'd'
alpha = 1    # Excitation fraction

# Experimental data has error estimate? (1 or 0)
has_S_err = 1
has_dS_err = 1

# Downsample?
num_q_down_to = 75 # Apply decimate() to downsample the curve to less than this number of points.

# Upper and lower bound of q
use_diff_q_range = 1 # Simply use q range from difference file; overwrites ql and qu below
                     # It is however useful to set the ql and qu in case of fallback. 
ql = 0.03
qu = 0.7

# Solvent electron density (rho, for pure water at 20 deg C it is 0.334)
rho = 0.334 * 0.98545 # 55.5 deg C density

# k chi - weighing factor
k_chi = 5e-7

# number of raster points to determine surface area (better be power of 2)
num_raster = 512

# Snapshots per delta_t steps for exponential moving averaging
# Memory time tau steps
delta_t = 50
tau = 5000


##### The rest you probably don't need to modify #####


# Parse files

with open(fpsf) as f:
    PSF = f.readlines()

#PSF = [x.strip() for x in PSF]
## Read bonds

get_bonds_from_now_on = 0
get_types_from_now_on = 0

for x in PSF:
    if get_bonds_from_now_on:
        #print(idx)
        temp_bonds = re.findall('\d+', x)
        temp_bonds = map(int, temp_bonds)
        while (idx < NBOND):
            bonds[idx][:] = temp_bonds[0:2]
            #print(bonds[idx][:])
            idx = idx + 1
            if (len(temp_bonds) > 2):
                temp_bonds[0:2] = []
            else:
                break
    if get_types_from_now_on:
        idx = int(re.search(r'\d+', x).group())
#        print(idx)
#        print(x)
        if x[24] == 'H':
            Ele[idx-1] = 0
            #print('Hydrogen')
        elif x[24] == 'C':
            Ele[idx-1] = 1 
        elif x[24] == 'N':
            Ele[idx-1] = 2 
        elif x[24] == 'O':
            Ele[idx-1] = 3 
        elif x[24] == 'S':
            Ele[idx-1] = 4 
        elif x[24:26] == 'Fe':
            Ele[idx-1] = 5
#        print(Ele[idx-1]) 
        if (idx == NATOM):
            print('Recorded all atoms.')
            get_types_from_now_on = 0

    if '!NBOND:' in x:
        print('NBONDS found.')
        print(x)
        print(re.search(r'\d+', x).group())
        NBOND = int(re.search(r'\d+', x).group())
        print('There are {:d} bonds.'.format(NBOND))
        get_bonds_from_now_on = 1
        idx = 0
        bonds = np.zeros((NBOND,2),dtype=int)

    if '!NATOM' in x:
        NATOM = int(re.search(r'\d+', x).group())
        if NATOM != num_atom:
            print('!!!!!! NATOM is larger than num_atom, check if this is a solvated model or you typed wrong num_atom!')
        print(NATOM)
        atoms = np.zeros((NATOM,3))
        Ele = np.zeros((NATOM,1),dtype=int)
        get_types_from_now_on = 1

print(bonds)
bonds = bonds.flatten()
print(Ele)


# Read PDB
with open(fpdb) as f:
    PDB = f.readlines()

PDB = [x.strip().split() for x in PDB]

## Read coordinates
for x in PDB:
    if x[0] == 'ATOM':
        atoms[int(x[1])-1][:] = x[6:9]

print(atoms)


HC = 0
HN = 0
HO = 0
HS = 0
num_ele = 5

for idx, atom in enumerate(Ele):
    if atom == 0:
#        print('Idx is {:d}'.format(idx))
        atom_H = bonds.tolist().index(idx+1)
#        print('atom_H index is {:d}'.format(atom_H))
        if (atom_H % 2 == 0):
            atom_X = atom_H + 1
        else:
            atom_X = atom_H - 1
#        print('atom_X index is {:d}'.format(atom_X))
        if (Ele[bonds[atom_X]-1] == 1):
            A = 'Carbon'
            HC = HC + 1
        elif (Ele[bonds[atom_X]-1] == 2):
            A = 'Nitrogen'
            HN = HN + 1
        elif (Ele[bonds[atom_X]-1] == 3):
            A = 'Oxygen'
            HO = HO + 1
        elif (Ele[bonds[atom_X]-1] == 4):
            A = 'Sulfur'
            HS = HS + 1

#        print('Corresponding heavy atom is {:d} ({:s})'.format(Ele[bonds[atom_X]-1],A))
        Ele[idx] = Ele[idx] + num_ele + Ele[bonds[atom_X]-1]

#print(Ele.tolist())

## Print things

with open(data_path + 'mol_param.hh','w') as f:
    f.write('\n')
    f.write('extern int Ele[{:d}];\n'.format(NATOM))
    f.write('extern int num_atom;\n')
    f.write('extern int num_atom2;\n')

with open(data_path + 'mol_param.cu','w') as f:
    f.write('\n#include "mol_param.hh"\n\n')
    f.write('int num_atom = {:d};\n'.format(NATOM))
    f.write('int num_atom2 = {:d};\n\n'.format(next_2048(NATOM)))
    f.write('int Ele[{:d}] = {{'.format(NATOM))
    f.write(', '.join(map(str,Ele.flatten())))
    f.write('};\n')
 
with open(data_path + 'coord_ref.hh','w') as f:
    f.write('\nextern float coord_ref[{:d}];\n'.format(3 * NATOM))
    f.write('extern float coord_init[{:d}];\n'.format(3 * NATOM))
    
with open(data_path + 'coord_ref.cu','w') as f:
    f.write('\n#include "coord_ref.hh"\n\n')
    f.write('float coord_ref[{:d}] = {{'.format(3 * NATOM))
    f.write(', '.join(map(str,atoms.flatten())))
    f.write('};\n')
    f.write('float coord_init[{:d}] = {{'.format(3 * NATOM))
    f.write(', '.join(map(str,atoms.flatten())))
    f.write('};\n')


# read expt files

if num_dS == 0:
    use_diff_q_range = 0
    print('There is no dS file, so we will not use q range from it.')
    print('Preset ql {:.3f}/A and qu {:.3f}/A will be used'.format(ql,qu))

S_exp = np.loadtxt(S_exp_file)
S_exp = np.array(S_exp)

# Now we deal with difference curve. Load file. 
if num_dS > 0:
    dS_exp = np.loadtxt(dS_exp_file)
    dS_exp = np.array(dS_exp)
else:
    dS_exp = S_exp
    use_diff_q_range = 0

print(dS_exp)
if use_diff_q_range == 1:
    ql = dS_exp[0,0]
    qu = dS_exp[-1,0]
    print('Using q range from dS file, which is {:.3f}/A to {:.3f}/A'.format(ql,qu))


if q_unit == 'nm':
    dS_exp[:,0] = np.divide(dS_exp[:,0],10.0)

if q_unit == 'nm':
    S_exp[:,0] = np.divide(S_exp[:,0],10.0)

num_q = len(S_exp[:,0])
num_q = len(S_exp[(S_exp[:,0]>ql) & (S_exp[:,0]<qu),0])
if num_q > num_q_down_to:
    down_sample_factor = np.ceil(num_q / num_q_down_to).astype(int)
    #print(S_exp)
    #print(down_sample_factor)
    print('Downsampling from {:d} q points to desired ({:d} points)'.format(num_q, num_q_down_to))
    S_exp = sig.decimate(S_exp[(S_exp[:,0]>ql) & (S_exp[:,0]<qu),:], down_sample_factor, axis=0)
else:
    print('There is no need to decimate. Number of q points ({:d}) is smaller than desired ({:d})'.format(num_q, num_q_down_to))
    try: 
        qidxl = np.asscalar(np.argwhere(S_exp[:,0] < ql)[-1])
        print(np.argwhere(S_exp[:,0] <ql))
    except:
        qidxl = 0
        print('There is no points with q smaller than {:.3f}'.format(ql))
    try:
        qidxu = np.asscalar(np.argwhere(S_exp[:,0] > qu)[0])
    except:
        qidxu = len(S_exp[:,0])
        print('There is no points with q larger than {:.3f}'.format(qu))
    #print(qidxl)
    #print(qidxu)
    S_exp = S_exp[qidxl:qidxu,:]

num_q = len(S_exp[:,0])
print(num_q)

print(S_exp)

# Since S_exp may be decimated, we interpolate dS_exp
x = dS_exp[:,0]
y = dS_exp[:,1]
f1 = interpolate.interp1d(x,y)
ynew = f1(S_exp[:,0])
if has_dS_err: 
    y2 = dS_exp[:,2]
    f2 = interpolate.interp1d(x,y2)
    y2new = f2(S_exp[:,0])

#dS_exp = []
#dS_exp.append(S_exp[:,0])
#dS_exp.append(ynew)
#dS_exp.append(y2new)
#print(dS_exp)


#print(S_exp)
# write expt_data.cu
with open(data_path + 'expt_data.cu','w') as f:
    f.write('#include \"expt_data.hh\"\n\n')
    f.write('int num_q = {:d};\n'.format(num_q))
    f.write('int num_q2 = {:d};\n'.format((num_q+31)/32*32))
    f.write('float q[{:d}] = {{'.format(num_q))
    f.write(', '.join(map(str,S_exp[:,0])))
    f.write('};\n')
    f.write('float S_exp[{:d}] = {{'.format(num_q))
    f.write(', '.join(map(str,S_exp[:,1])))
    f.write('};\n')
    if has_S_err:
        f.write('float S_err[{:d}] = {{'.format(num_q))
        f.write(', '.join(map(str,S_exp[:,2])))
        f.write('};\n')
    else:
        f.write('float S_err[{:d}] = {{'.format(num_q))
        f.write(', '.join(map(str,np.ones_like(S_exp[:,0]))))
        f.write('};\n')
    if num_dS > 0:
        f.write('float dS_exp[{:d}] = {{'.format(num_q))
        f.write(', '.join(map(str,ynew)))
        f.write('};\n')
    else: 
        f.write('float dS_exp[{:d}] = {{'.format(num_q))
        f.write(', '.join(map(str,np.zeros_like(S_exp[:,0]))))
        f.write('};\n\n')
    if has_dS_err:
        f.write('float dS_err[{:d}] = {{'.format(num_q))
        f.write(', '.join(map(str,y2new)))
        f.write('};\n\n')
    else: 
        f.write('float dS_err[{:d}] = {{'.format(num_q))
        f.write(', '.join(map(str,np.ones_like(S_exp[:,0]))))
        f.write('};\n\n')
    f.write('int has_S_err = {:d};\n'.format(has_S_err))
    f.write('int has_dS_err = {:d};\n'.format(has_dS_err))
    
with open(data_path + 'expt_data.hh','w') as f:
    f.write('extern int num_q;\n');
    f.write('extern int num_q2;\n');
    f.write('extern float q[{:d}];\n'.format(num_q))
    f.write('extern float S_exp[{:d}];\n'.format(num_q))
    f.write('extern float S_err[{:d}];\n'.format(num_q))
    f.write('extern float dS_exp[{:d}];\n'.format(num_q))
    f.write('extern float dS_err[{:d}];\n'.format(num_q))
    f.write('extern int has_S_err;\n')
    f.write('extern int has_dS_err;\n')

# write env_param

with open(data_path + 'env_param.cu','w') as f:
    f.write('#include \"env_param.hh\"\n\n')
    f.write('float k_chi = {:.3e};\n'.format(k_chi))
    f.write('int num_ele = 6;\n')
    f.write('int num_raster = {:d};\n'.format(num_raster))
    f.write('int num_raster2 = {:d};\n'.format(num_raster))
    f.write('float sol_s = 1.80;\n');
    f.write('float vdW[7] = {1.07, 1.58, 0.84, 1.30, 1.68, 1.24, 1.67};\n');
    f.write('float c2_H[10] = { 0.00000, -0.08428, -0.68250,  1.59535,  0.23293,  0.00000, \n')
    f.write('                   1.86771,  3.04298,  4.06575,  0.79196};\n')
    f.write('float r_m = 1.62;\n')
    f.write('float rho = {:.4f};\n'.format(rho))
    f.write('int delta_t = {:d};\n'.format(delta_t))
    f.write('int tau = {:d};\n'.format(tau))


with open(data_path + 'env_param.hh','w') as f:
    f.write('extern float k_chi;\n')
    f.write('extern int num_ele;\n')
    f.write('extern int num_raster;\n')
    f.write('extern int num_raster2;\n')
    f.write('extern float sol_s;\n');
    f.write('extern float vdW[7];\n');
    f.write('extern float c2_H[10];\n')
    f.write('extern float r_m;\n')
    f.write('extern float rho;\n')
    f.write('extern int delta_t;\n')
    f.write('extern int tau;\n')

