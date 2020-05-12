#include "env_param.hh"

float k_chi = 1.000e-05; // Gives about 90 kcal/mol XSP.
int num_ele = 6;         // Number of different elements (todo)
int num_raster = 512;    // Number of raster points to sample when calculating SASA
int num_raster2 = 512;   // Above rounded up to 512
float sol_s = 1.80;      // Solvent radius in Angstrom.

// vdW radii for H, C, N, O, S, Fe, water (from Svergun 1995 CRYSOL paper)
float vdW[7] = {1.07, 1.58, 0.84, 1.30, 1.68, 1.24, 1.67};

// Currently not used, the c2 for different atom types estimated using HyPred data.
float c2_H[10] = { 0.00000, -0.08428, -0.68250,  1.59535,  0.23293,  0.00000, 
                   1.86771,  3.04298,  4.06575,  0.79196};

float r_m = 1.62;  // Average atomic radius for form factor calculation, 
                   // see Svergun 1995 CRYSOL paper.

float rho = 0.334; // Solvent electron density in e- / A^3. 
                   // 0.334 is for water at 20 C.

int delta_t = 50;  // MD steps, 100 fs. For exponential averaging.
int tau = 5e4;     // MD steps, 100 ps.


