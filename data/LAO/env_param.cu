
#include "env_param.hh"

float k_chi = 1.190e-7; // Gives 10 kcal/mol XSP
int num_ele = 6;
int num_raster = 512;
int num_raster2 = 512;
float sol_s = 1.80;
float vdW[7] = {1.07, 1.58, 0.84, 1.30, 1.68, 1.24, 1.67};
float c2_H[10] = { 0.00000, -0.08428, -0.68250,  1.59535,  0.23293,  0.00000, 
                   1.86771,  3.04298,  4.06575,  0.79196};
float r_m = 1.62;
float rho = 0.334;
int delta_t = 50; // Frames, 100 fs.
int tau = 5e3; // Frames, 10 ps.


