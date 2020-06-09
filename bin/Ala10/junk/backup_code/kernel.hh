



//__global__ void scat_calc (float *coord, float *Force, int *Ele, float *FF, float *q, float *S_ref, float *dS, float *S_calc, int num_atom, int num_q, int num_ele, float *Aq, float alpha, float k_chi, float sigma2, float *f_ptxc, float *f_ptyc, float *f_ptzc, float *S_calcc, int num_atom2, int num_q2);
__global__ void dist_calc (float *coord, //float *dx, float *dy, float *dz, 
                           int *cloase_num, int *close_flag, int *close_idx, int num_atom, int num_atom2);
__global__ void pre_scan_close (int *close_flag, int *close_num, int *close_idx, int num_atom2);
__global__ void surf_calc (float *coord, int *Ele, //float *r2, 
                           int *close_num, int *close_flag, int *close_idx, float *vdW, int num_atom, int num_atom2, int num_raster, float sol_s, float *V);
__global__ void surf_calc_surf_grad (float *coord, int *Ele, //float *r2, 
                           int *close_num, int *close_flag, int *close_idx, float *vdW, int num_atom, int num_atom2, int num_raster, float sol_s, float *V,
                           float *surf_grad, float offset);
__global__ void border_scat (float *coord, int *Ele, float *r2, float *raster, float *V, int num_atom, int num_atom2, int num_raster);
__global__ void V_calc (float *V, int num_atom2);
__global__ void scat_calc (float *coord,  float *Force,   int *Ele,      float *WK,     float *q_S_ref_dS, 
                           float *S_calc, int num_atom,   int num_q,     int num_ele,   float *Aq, 
                           float alpha,   float k_chi,    float sigma2,  float *f_ptxc, float *f_ptyc, 
                           float *f_ptzc, float *S_calcc, int num_atom2, int num_q2,    float *vdW, 
                           float c1,      float c2,       float *V,      float r_m);

__global__ void force_calc (float *Force, int num_atom, int num_q, float *f_ptxc, float *f_ptyc, float *f_ptzc, int num_atom2, int num_q2); 
/*__global__ void force_proj (float *coord, float *Force, float *rot, float *rot_pt, int *bond_pp, int num_pp, int num_atom, int num_atom2);
__global__ void pp_assign (float *coord, float *Force, float *rot, int *bond_pp, int num_pp, int num_atom);
__device__ float dot (float a1, float a2, float a3, float b1, float b2, float b3);
__device__ float cross2 (float a2, float a3, float b2, float b3);*/
