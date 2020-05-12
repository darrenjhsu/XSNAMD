
#include "mol_param.hh"

int num_atom = 104;   // number of atoms
int num_atom2 = 2048; // number of atoms rounded up to 2048 (for CUDA reasons)

// Element lists for vdW radii and form factors.
int Ele[104] = {2, 7, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 7};
