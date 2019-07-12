
#include "mol_param.hh"

int num_atom = 1932;
int num_atom2 = 2048;

int Ele[1932] = {2, 7, 7, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 3, 8, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 4, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 1, 2, 7, 7, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 1, 6, 1, 3, 8, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 6, 1, 3, 2, 7, 1, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 3, 8, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 1, 6, 6, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 2, 7, 1, 1, 1, 6, 1, 6, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 4, 1, 3, 2, 7, 1, 6, 1, 6, 3, 8, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 3, 8, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 2, 7, 1, 1, 6, 2, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 3, 8, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 3, 8, 1, 3, 2, 7, 1, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 1, 6, 1, 3, 8, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 3, 8, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 3, 8, 1, 3, 2, 7, 1, 6, 1, 6, 3, 8, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 1, 6, 1, 3, 8, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 2, 7, 1, 1, 1, 6, 1, 6, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 4, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 1, 6, 6, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 2, 7, 1, 1, 6, 2, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 3, 8, 1, 3, 2, 7, 1, 6, 1, 6, 6, 3, 8, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 4, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 3, 8, 1, 3, 2, 7, 1, 6, 1, 6, 6, 4, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 3, 8, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 4, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 4, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 1, 6, 1, 3, 8, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 2, 7, 1, 1, 1, 6, 1, 6, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 2, 7, 1, 1, 6, 2, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 4, 1, 3, 2, 7, 1, 6, 1, 6, 6, 3, 8, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 2, 7, 7, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 1, 6, 2, 7, 1, 1, 1, 6, 1, 6, 1, 6, 1, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 6, 6, 1, 3, 2, 7, 1, 6, 1, 6, 6, 4, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 3, 3, 1, 3, 2, 7, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 2, 7, 7, 7, 1, 3, 3, 1, 3, 8, 8, 3, 8, 8};