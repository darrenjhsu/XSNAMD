
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "env_param.hh"
//#include "scat_param.hh"
#include "mol_param.hh"
#include "WaasKirf.hh"
#define PI 3.14159265359
//#include <cuda_fp16.h>


__global__ void dist_calc (
    float *coord, 
    int *close_num,
    int *close_flag, 
    int *close_idx, 
    int num_atom, 
    int num_atom2) {

    // close_flag is a 1024 x num_atom2 int matrix initialized to 0.
    // close_idx: A num_atom x 200 int matrix, row i of which only the first close_num[i] elements are defined. (Otherwise it's -1). 
    __shared__ float x_ref, y_ref, z_ref;
    __shared__ int idz;
    __shared__ int temp[2048];
    // Calc distance
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        if (threadIdx.x == 0) {
            x_ref = coord[3*ii  ];
            y_ref = coord[3*ii+1];
            z_ref = coord[3*ii+2];
        }
        int idy = ii % gridDim.x; // This will be what row of close_flag this block is putting its value in.
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            float r2t = (coord[3*jj  ] - x_ref) * (coord[3*jj  ] - x_ref) + 
                        (coord[3*jj+1] - y_ref) * (coord[3*jj+1] - y_ref) + 
                        (coord[3*jj+2] - z_ref) * (coord[3*jj+2] - z_ref); 
 
            if (r2t < 34.0) {
                close_flag[idy*num_atom2+jj] = 1; // roughly 2 A + 2 A vdW + 2 * 1.8 A probe
            } else { 
                close_flag[idy*num_atom2+jj] = 0;
            }
            if (ii == jj) close_flag[idy*num_atom2+jj] = 0;
        }
        __syncthreads();
        // Do pre scan
        idz = 0;
        int temp_sum = 0;
        for (int jj = threadIdx.x; jj < num_atom2; jj += 2 * blockDim.x) {
            int idx = jj % blockDim.x; 
            int offset = 1;
            temp[2 * idx]     = close_flag[idy * num_atom2 + 2 * blockDim.x * idz + 2 * idx];
            temp[2 * idx + 1] = close_flag[idy * num_atom2 + 2 * blockDim.x * idz + 2 * idx + 1];
            for (int d = 2 * blockDim.x>>1; d > 0; d >>= 1) { // up-sweep
                __syncthreads();
                if (idx < d) {
                    int ai = offset * (2 * idx + 1) - 1;
                    int bi = offset * (2 * idx + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }
            __syncthreads();
            temp_sum = close_num[ii];
            __syncthreads();
            if (idx == 0) {
                close_num[ii] += temp[2 * blockDim.x - 1]; // log the total number of 1's in this blockDim
                temp[2 * blockDim.x - 1] = 0;
            }
            __syncthreads();
            for (int d = 1; d < blockDim.x * 2; d *= 2) { //down-sweep
                offset >>= 1;
                __syncthreads();
                if (idx < d) {
                    int ai = offset * (2 * idx + 1) - 1;
                    int bi = offset * (2 * idx + 2) - 1;
                    int t    = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
        
            __syncthreads();
        
            // Finally assign the indices
            if (close_flag[idy * num_atom2 + 2 * blockDim.x * idz + 2 * idx] == 1) {
                close_idx[ii * 1024 + temp[2*idx] + temp_sum] = 2 * idx + 2 * blockDim.x * idz;
            }
            if (close_flag[idy * num_atom2 + 2 * blockDim.x * idz + 2 * idx + 1] == 1) {
                close_idx[ii * 1024 + temp[2*idx+1] + temp_sum] = 2*idx+1 + 2 * blockDim.x * idz;
            }
            idz++;
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(512,4) surf_calc_surf_grad (
    float *coord, 
    int *Ele, 
    int *close_num, 
    int *close_idx, 
    float *vdW, 
    int num_atom, 
    int num_atom2, 
    int num_raster, 
    float sol_s, 
    float *V,
    float *surf_grad,
    float offset) {

    // num_raster should be a number of 2^n. 
    // sol_s is solvent radius (default = 1.8 A)
    __shared__ float vdW_s; // vdW radius of the center atom
    __shared__ int pts[512]; // All spherical raster points
    __shared__ int ptspx[512], ptsmx[512], ptspy[512], ptsmy[512], ptspz[512], ptsmz[512];
    __shared__ float L, r;
    
    if (blockIdx.x >= num_atom) return;
    L = sqrt(num_raster * PI);
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        int atom1t = Ele[ii];
        if (atom1t > 5) atom1t = 0;
        vdW_s = vdW[atom1t];
        r = vdW_s + sol_s;
        for (int jj = threadIdx.x; jj < num_raster; jj += blockDim.x) {
            int pt = 1;
            int ptpx = 1;
            int ptmx = 1;
            int ptpy = 1;
            int ptmy = 1;
            int ptpz = 1;
            int ptmz = 1;
            
            float h = 1.0 - (2.0 * (float)jj + 1.0) / (float)num_raster;
            float p = acos(h);
            float t = L * p; 
            float xu = sin(p) * cos(t);
            float yu = sin(p) * sin(t);
            float zu = cos(p);
            // vdW points
            float x = vdW_s * xu + coord[3*ii];
            float y = vdW_s * yu + coord[3*ii+1];
            float z = vdW_s * zu + coord[3*ii+2];
            // Solvent center
            float x2 = r * xu + coord[3*ii];
            float y2 = r * yu + coord[3*ii+1];
            float z2 = r * zu + coord[3*ii+2];
            for (int kk = 0; kk < close_num[ii]; kk++) {
                int atom2i = close_idx[ii * 1024 + kk];
                int atom2t = Ele[atom2i];
                if (atom2t > 5) atom2t = 0;
                float dx = (x - coord[3*atom2i]);
                float dy = (y - coord[3*atom2i+1]);
                float dz = (z - coord[3*atom2i+2]);
                float dx2 = (x2 - coord[3*atom2i]);
                float dy2 = (y2 - coord[3*atom2i+1]);
                float dz2 = (z2 - coord[3*atom2i+2]);
                float dr2 = dx * dx + dy * dy + dz * dz; 
                float dr22 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                // vdW points must not cross into other atom
                if (dr2 < vdW[atom2t] * vdW[atom2t]) pt = 0; //pts[jj] = 0;
                // solvent center has to be far enough
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) pt = 0; //pts[jj] = 0;

                // Plus x
                dr2 =  (dx + offset)  * (dx + offset) + dy * dy + dz * dz;
                dr22 = (dx2 + offset) * (dx2 + offset) + dy2 * dy2 + dz2 * dz2;
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptpx = 0; //ptspx[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptpx = 0; //ptspx[jj] = 0;
                // Minus x
                dr2 =  (dx - offset)  * (dx - offset) + dy * dy + dz * dz;
                dr22 = (dx2 - offset) * (dx2 - offset) + dy2 * dy2 + dz2 * dz2;
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptmx = 0; //ptsmx[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptmx = 0; //ptsmx[jj] = 0;
                // Plus y
                dr2 =  dx * dx   + (dy + offset)  * (dy + offset) + dz * dz; 
                dr22 = dx2 * dx2 + (dy2 + offset) * (dy2 + offset) + dz2 * dz2;
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptpy = 0; //ptspy[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptpy = 0; //ptspy[jj] = 0;
                // Minus y
                dr2 =  dx * dx   + (dy - offset)  * (dy - offset) + dz * dz; 
                dr22 = dx2 * dx2 + (dy2 - offset) * (dy2 - offset) + dz2 * dz2;
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptmy = 0; //ptsmy[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptmy = 0; //ptsmy[jj] = 0;
                // Plus z
                dr2 =  dx * dx + dy * dy + (dz + offset) * (dz + offset); 
                dr22 = dx2 * dx2 + dy2 * dy2 + (dz2 + offset) * (dz2 + offset);
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptpz = 0; //ptspz[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptpz = 0; //ptspz[jj] = 0;
                // Minus z
                dr2 =  dx * dx + dy * dy + (dz - offset) * (dz - offset); 
                dr22 = dx2 * dx2 + dy2 * dy2 + (dz2 - offset) * (dz2 - offset);
                if (dr2 < vdW[atom2t] * vdW[atom2t]) ptmz = 0; //ptsmz[jj] = 0;
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) ptmz = 0; //ptsmz[jj] = 0;
                
            }
            pts[jj] = pt;
            ptspx[jj] = ptpx;
            ptsmx[jj] = ptmx;
            ptspy[jj] = ptpy;
            ptsmy[jj] = ptmy;
            ptspz[jj] = ptpz;
            ptsmz[jj] = ptmz;
        }
        // Sum pts == 1, calc surf area and assign to V[ii]
        for (int stride = num_raster / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                pts[iAccum] += pts[stride + iAccum];

                ptspx[iAccum] += ptspx[stride + iAccum];
                ptsmx[iAccum] += ptsmx[stride + iAccum];
                ptspy[iAccum] += ptspy[stride + iAccum];
                ptsmy[iAccum] += ptsmy[stride + iAccum];
                ptspz[iAccum] += ptspz[stride + iAccum];
                ptsmz[iAccum] += ptsmz[stride + iAccum];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            V[ii] = (float)pts[0]/(float)num_raster;// * 4.0 * r * r * PI ;

            surf_grad[3*ii  ] = (float)(ptspx[0] - ptsmx[0]) / 2.0 / offset / (float)num_raster;
            surf_grad[3*ii+1] = (float)(ptspy[0] - ptsmy[0]) / 2.0 / offset / (float)num_raster;
            surf_grad[3*ii+2] = (float)(ptspz[0] - ptsmz[0]) / 2.0 / offset / (float)num_raster;
        }
    }
}

__global__ void __launch_bounds__(512,4) surf_calc (
    float *coord, 
    int *Ele, 
    int *close_num, 
    int *close_idx, 
    float *vdW, 
    int num_atom, 
    int num_atom2, 
    int num_raster, 
    float sol_s, 
    float *V) {

    // num_raster should be a number of 2^n. 
    // sol_s is solvent radius (default = 1.8 A)
    __shared__ float vdW_s; // vdW radius of the center atom
    __shared__ int pts[512]; // All spherical raster points
    __shared__ float L, r;
    
    if (blockIdx.x >= num_atom) return;
    L = sqrt(num_raster * PI);
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        int atom1t = Ele[ii];
        if (atom1t > 5) atom1t = 0;
        vdW_s = vdW[atom1t];
        r = vdW_s + sol_s;
        for (int jj = threadIdx.x; jj < num_raster; jj += blockDim.x) {
            int pt = 1;
            
            float h = 1.0 - (2.0 * (float)jj + 1.0) / (float)num_raster;
            float p = acos(h);
            float t = L * p; 
            float xu = sin(p) * cos(t);
            float yu = sin(p) * sin(t);
            float zu = cos(p);
            // vdW points
            float x = vdW_s * xu + coord[3*ii];
            float y = vdW_s * yu + coord[3*ii+1];
            float z = vdW_s * zu + coord[3*ii+2];
            // Solvent center
            float x2 = r * xu + coord[3*ii];
            float y2 = r * yu + coord[3*ii+1];
            float z2 = r * zu + coord[3*ii+2];
            for (int kk = 0; kk < close_num[ii]; kk++) {
                int atom2i = close_idx[ii * 1024 + kk];
                int atom2t = Ele[atom2i];
                if (atom2t > 5) atom2t = 0;
                float dx = (x - coord[3*atom2i]);
                float dy = (y - coord[3*atom2i+1]);
                float dz = (z - coord[3*atom2i+2]);
                float dx2 = (x2 - coord[3*atom2i]);
                float dy2 = (y2 - coord[3*atom2i+1]);
                float dz2 = (z2 - coord[3*atom2i+2]);
                float dr2 = dx * dx + dy * dy + dz * dz; 
                float dr22 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                // vdW points must not cross into other atom
                if (dr2 < vdW[atom2t] * vdW[atom2t]) pt = 0; //pts[jj] = 0;
                // solvent center has to be far enough
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) pt = 0; //pts[jj] = 0;
                
            }
            pts[jj] = pt;
        }
        // Sum pts == 1, calc surf area and assign to V[ii]
        for (int stride = num_raster / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                pts[iAccum] += pts[stride + iAccum];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            V[ii] = (float)pts[0]/(float)num_raster;// * 4.0 * r * r * PI ;
        }
    }
}


__global__ void sum_V (
    float *V, 
    float *V_s, 
    int num_atom, 
    int num_atom2, 
    int *Ele, 
    float *vdW) {

    for (int ii = threadIdx.x; ii < num_atom2; ii += blockDim.x) {
        if (ii < num_atom) {
            int atomi = Ele[ii];
            if (atomi > 5) atomi = 0;
            V_s[ii] = V[ii] * 4.0 * PI * vdW[atomi] * vdW[atomi];
        } else {
            V_s[ii] = 0.0;
        }
    }
    for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
            V_s[iAccum] += V_s[stride + iAccum];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) printf("Convex contact area = %.3f A^2.\n", V_s[0]);
}

__global__ void FF_calc (
    float *q_S_ref_dS, 
    float *WK, 
    float *vdW, 
    int num_q, 
    int num_ele, 
    float c1, 
    float r_m, 
    float *FF_table,
    float rho) {

    __shared__ float q_pt, q_WK, C1, expC1;
    __shared__ float FF_pt[7]; // num_ele + 1, the last one for water.
    __shared__ float vdW_s[7];
    __shared__ float WK_s[66]; 
    __shared__ float C1_PI_43_rho;
    if (blockIdx.x >= num_q) return; // out of q range
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
        q_pt = q_S_ref_dS[ii];
        q_WK = q_pt / 4.0 / PI;
        // FoXS C1 term
        expC1 = -powf(4.0 * PI / 3.0, 1.5) * q_WK * q_WK * r_m * r_m * (c1 * c1 - 1.0) / 4.0 / PI;
        C1 = powf(c1,3) * exp(expC1);
        C1_PI_43_rho = C1 * PI * 4.0 / 3.0 * rho;
        for (int jj = threadIdx.x; jj < 11 * num_ele; jj += blockDim.x) {
            WK_s[jj] = WK[jj];
        }
        __syncthreads();

        // Calculate Form factor for this block (or q vector)
        for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
            vdW_s[jj] = vdW[jj];
            if (jj == num_ele) {
                // water
                FF_pt[jj] = WK_s[3*11+5];
                FF_pt[jj] += 2.0 * WK_s[5];
                FF_pt[jj] -= C1_PI_43_rho * powf(vdW_s[jj],3.0) * exp(-PI * vdW_s[jj] * vdW_s[jj] * q_WK * q_WK);
                for (int kk = 0; kk < 5; kk ++) {
                    FF_pt[jj] += WK_s[3*11+kk] * exp(-WK_s[3*11+kk+6] * q_WK * q_WK);
                    FF_pt[jj] += WK_s[kk] * exp(-WK_s[kk+6] * q_WK * q_WK);
                    FF_pt[jj] += WK_s[kk] * exp(-WK_s[kk+6] * q_WK * q_WK);
                }
            } else { 
                FF_pt[jj] = WK_s[jj*11+5];
                // The part is for excluded volume
                FF_pt[jj] -= C1_PI_43_rho * powf(vdW_s[jj],3.0) * exp(-PI * vdW_s[jj] * vdW_s[jj] * q_WK * q_WK);
                for (int kk = 0; kk < 5; kk++) {
                    FF_pt[jj] += WK_s[jj*11+kk] * exp(-WK_s[jj*11+kk+6] * q_WK * q_WK); 
                }
            }
            FF_table[ii*(num_ele+1)+jj] = FF_pt[jj];
        }
    }
}

__global__ void create_FF_full_HyPred (
    float *FF_table, 
    float *V, 
    float c2_F,
    float *c2_H,
    int *Ele, 
    float *FF_full, 
    int num_q, 
    int num_ele, 
    int num_atom, 
    int num_atom2) {
    
    __shared__ float FF_pt[7];
    __shared__ float hydration[10];
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        // Get form factor for this block (or q vector)
        if (ii < num_q) {
            for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
                FF_pt[jj] = FF_table[ii*(num_ele+1)+jj];
            }
        }
        __syncthreads();

        for (int jj = threadIdx.x; jj < 10; jj += blockDim.x) {
            hydration[jj] = c2_F * c2_H[jj] * FF_pt[num_ele];
        }
        __syncthreads();
        
        // Calculate atomic form factor for this q
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            int atomt = Ele[jj];
            if (atomt > 5) {  // Which means this is a hydrogen
                FF_full[ii*num_atom2 + jj] = FF_pt[0];
                FF_full[ii*num_atom2 + jj] += hydration[atomt] * V[jj];
            } else { // Heavy atoms - do the same as before
                FF_full[ii*num_atom2 + jj] = FF_pt[atomt];
                FF_full[ii*num_atom2 + jj] += hydration[atomt] * V[jj];
            }
        }
    }
}

__global__ void create_FF_full_FoXS (
    float *FF_table, 
    float *V, 
    float c2, 
    int *Ele, 
    float *FF_full, 
    int num_q, 
    int num_ele, 
    int num_atom, 
    int num_atom2) {

    __shared__ float FF_pt[7];
    float hydration;
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        // Get form factor for this block (or q vector)
        if (ii < num_q) {
            for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
                FF_pt[jj] = FF_table[ii*(num_ele+1)+jj];
            }
        }
        __syncthreads();
        
        // In FoXS since c2 remains the same for all elements it is reduced to one value.
        hydration = c2 * FF_pt[num_ele];
        
        // Calculate atomic form factor for this q
        // However to keep compatible to HyPred method we leave atom type def unchanged.
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            int atomt = Ele[jj];
            if (atomt > 5) {  // Which means this is a hydrogen
                FF_full[ii*num_atom2 + jj] = FF_pt[0];
                FF_full[ii*num_atom2 + jj] += hydration * V[jj];
            } else {          // Heavy atoms - do the same as before
                FF_full[ii*num_atom2 + jj] = FF_pt[atomt];
                FF_full[ii*num_atom2 + jj] += hydration * V[jj];
            }
        }
    }
}

__global__ void create_FF_full_FoXS_surf_grad (
    float *FF_table, 
    float *V, 
    float c2, 
    int *Ele, 
    float *FF_full,
    float *surf_grad, 
    int num_q, 
    int num_ele, 
    int num_atom, 
    int num_atom2) {

    __shared__ float FF_pt[7];
    float hydration;
    for (int ii = blockIdx.x; ii < num_q+1; ii += gridDim.x) {

        // Get form factor for this block (or q vector)
        if (ii < num_q) {
            for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
                FF_pt[jj] = FF_table[ii*(num_ele+1)+jj];
            }
        }
        __syncthreads();
        
        // In FoXS since c2 remains the same for all elements it is reduced to one value.
        hydration = c2 * FF_pt[num_ele];
        //if (ii == num_q && threadIdx.x == 0) {printf("Hydration is: %6.3f\n", hydration);}
        __syncthreads();
        // Calculate atomic form factor for this q
        // However to keep compatible to HyPred method we leave atom type def unchanged.
        if (ii == num_q) {
            // calculate surf_grad
            for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
                //int atomt = Ele[jj];
                //printf("B surf grads = %6.3f, %6.3f, %6.3f. \n", 
                //       surf_grad[3*jj], surf_grad[3*jj+1], surf_grad[3*jj+2]);
                /*surf_grad[3*jj]   *= hydration;
                surf_grad[3*jj+1] *= hydration;
                surf_grad[3*jj+2] *= hydration;*/
                surf_grad[3*jj]   *= c2;
                surf_grad[3*jj+1] *= c2;
                surf_grad[3*jj+2] *= c2;
                //printf("A surf grads = %6.3f, %6.3f, %6.3f. \n", 
                //       surf_grad[3*jj], surf_grad[3*jj+1], surf_grad[3*jj+2]);
            }        
        } else {
            for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
                int atomt = Ele[jj];
                if (atomt > 5) {  // Which means this is a hydrogen
                    FF_full[ii*num_atom2 + jj] = FF_pt[0];
                    FF_full[ii*num_atom2 + jj] += hydration * V[jj];
                } else {          // Heavy atoms - do the same as before
                    FF_full[ii*num_atom2 + jj] = FF_pt[atomt];
                    FF_full[ii*num_atom2 + jj] += hydration * V[jj];
                }
            }
        }
        if (threadIdx.x == 0) FF_full[ii * num_atom2 + num_atom + 1] = FF_pt[num_ele];
    }
}


__global__ void __launch_bounds__(1024,2) scat_calc (
    float *coord, 
    int *Ele,
    float *q_S_ref_dS, 
    float *S_calc, 
    int num_atom,   
    int num_q,     
    int num_ele,   
    float *Aq, 
    float alpha,   
    float k_chi,    
    float *sigma2,  
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    float *S_calcc, 
    int num_atom2,
    float *FF_full) {

    float q_pt, sigma2_pt; 

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q_S_ref_dS[ii];
        sigma2_pt = sigma2[ii];

        // Calculate scattering for Aq
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            // for every atom jj
            float atom1x = coord[3*jj+0];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            float S_calccs = 0.0;
            float f_ptxcs = 0.0;
            float f_ptycs = 0.0;
            float f_ptzcs = 0.0;
            for (int kk = 0; kk < num_atom; kk++) {
                // for every atom kk
                float FF_kj = FF_full[ii * num_atom2 + jj] * FF_full[ii *num_atom2 + kk];
                if (q_pt == 0.0 || kk == jj) {
                    S_calccs += FF_kj;
                } else {
                    /*float dx = coord[3*kk+0] - atom1x;
                    float dy = coord[3*kk+1] - atom1y;
                    float dz = coord[3*kk+2] - atom1z;*/
                    float dx = atom1x - coord[3*kk+0];
                    float dy = atom1y - coord[3*kk+1];
                    float dz = atom1z - coord[3*kk+2];
                    float r = sqrt(dx*dx+dy*dy+dz*dz);
                    float qr = q_pt * r; 
                    float sqr = sin(qr) / qr;
                    float dsqr = cos(qr) - sqr;
                    float prefac = FF_kj * dsqr / r / r;
                    prefac += prefac;
                    S_calccs += FF_kj * sqr;
                    f_ptxcs += prefac * dx;
                    f_ptycs += prefac * dy;
                    f_ptzcs += prefac * dz;
                }
            }
            S_calcc[ii*num_atom2+jj] = S_calccs;
            f_ptxc[ii*num_atom2+jj] = f_ptxcs;
            f_ptyc[ii*num_atom2+jj] = f_ptycs;
            f_ptzc[ii*num_atom2+jj] = f_ptzcs;
        }
        
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        S_calc[ii] = S_calcc[ii * num_atom2];
        __syncthreads();


        if (threadIdx.x == 0) {
            Aq[ii] = S_calc[ii] - q_S_ref_dS[ii+num_q];
            Aq[ii] *= -alpha;
            Aq[ii] += q_S_ref_dS[ii + 2*num_q];
            Aq[ii] *= k_chi / sigma2_pt;
            Aq[ii] += Aq[ii];
        }
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
        }
    }
}

__global__ void __launch_bounds__(1024,2) scat_calc_surf_grad (
    float *coord, 
    int *Ele,
    float *q_S_ref_dS, 
    float *S_calc, 
    int num_atom,   
    int num_q,     
    int num_ele,   
    float *Aq, 
    float alpha,   
    float k_chi,    
    float *sigma2,  
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    float *S_calcc, 
    int num_atom2,
    float *surf_grad,
    float *FF_full) {

    float q_pt, sigma2_pt, FF_w; 

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q_S_ref_dS[ii];
        sigma2_pt = sigma2[ii];
        FF_w = FF_full[ii * num_atom2 + num_ele + 1]; // Water form factor at this q
        // Calculate scattering for Aq
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            // for every atom jj
            float atom1x = coord[3*jj+0];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            float S_calccs = 0.0;
            float f_ptxcs = 0.0;
            float f_ptycs = 0.0;
            float f_ptzcs = 0.0;
            float FF_j = FF_full[ii * num_atom2 + jj];
            for (int kk = 0; kk < num_atom; kk++) {
                float FF_kj = FF_j * FF_full[ii * num_atom2 + kk];
                float dx = atom1x - coord[3*kk+0];
                float dy = atom1y - coord[3*kk+1];
                float dz = atom1z - coord[3*kk+2];
                /*float dx = coord[3*kk+0] - atom1x;
                float dy = coord[3*kk+1] - atom1y;
                float dz = coord[3*kk+2] - atom1z;*/
                float r = sqrt(dx*dx+dy*dy+dz*dz);
                float prefac = 0.0;
                float sqr = 1.0;
                if (kk == jj) r = 1.0;
                if (kk == jj || q_pt == 0.0) {
                    sqr = 1.0;
                } else {
                    float qr = q_pt * r; 
                    sqr = sin(qr) / qr;
                    float dsqr = cos(qr) - sqr;
                    prefac *= sqr;
                    prefac += FF_kj * dsqr;
                }
                prefac += prefac;
                prefac = prefac / r / r;
                S_calccs += FF_kj * sqr;
                f_ptxcs += prefac * dx;
                f_ptycs += prefac * dy;
                f_ptzcs += prefac * dz;

                prefac = FF_j * sqr * FF_w;
                prefac += prefac;
                //printf("sqr = %6.3f, FF_j = %6.3f, prod = %6.3f \n", sqr, FF_j, sqr*FF_j);
                //printf("surf grads = %6.3f, %6.3f, %6.3f. \n", surf_grad[3*jj], surf_grad[3*jj+1], surf_grad[3*jj+2]);
                f_ptxcs += prefac * surf_grad[3*jj  ];
                f_ptycs += prefac * surf_grad[3*jj+1];
                f_ptzcs += prefac * surf_grad[3*jj+2];
            }
            S_calcc[ii*num_atom2+jj] = S_calccs;
            f_ptxc[ii*num_atom2+jj] = f_ptxcs;
            f_ptyc[ii*num_atom2+jj] = f_ptycs;
            f_ptzc[ii*num_atom2+jj] = f_ptzcs;
        }
        
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        S_calc[ii] = S_calcc[ii * num_atom2];
        __syncthreads();


        if (threadIdx.x == 0) {
            Aq[ii] = S_calc[ii] - q_S_ref_dS[ii+num_q];
            Aq[ii] *= -alpha;
            Aq[ii] += q_S_ref_dS[ii + 2*num_q];
            Aq[ii] *= k_chi / sigma2_pt;
            Aq[ii] += Aq[ii];
        }
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
        }
    }
}

__global__ void __launch_bounds__(1024,2) scat_calc_EMA (
    float *coord, 
    int *Ele,
    float *q_S_ref_dS, 
    double *S_calc, 
    int num_atom,   
    int num_q,     
    int num_ele,   
    float *Aq, 
    float alpha,   
    float k_chi,    
    float *sigma2,  
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    float *S_calcc, 
    int num_atom2,
    float *FF_full,
    double *S_old,
    double EMA_norm) {

    // EMA_norm is computed on the host. See Chen & Hub, Biophysics 2015 2573-2584.

    float q_pt, sigma2_pt; 

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q_S_ref_dS[ii];
        sigma2_pt = sigma2[ii];
        // Calculate scattering for Aq
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            // for every atom jj
            float atom1x = coord[3*jj+0];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            float S_calccs = 0.0;
            float f_ptxcs = 0.0;
            float f_ptycs = 0.0;
            float f_ptzcs = 0.0;
            for (int kk = 0; kk < num_atom; kk++) {
                // for every atom kk
                float FF_kj = FF_full[ii * num_atom2 + jj] * FF_full[ii *num_atom2 + kk];
                if (q_pt == 0.0 || kk == jj) {
                    S_calccs += FF_kj;
                } else {
                    /*float dx = coord[3*kk+0] - atom1x;
                    float dy = coord[3*kk+1] - atom1y;
                    float dz = coord[3*kk+2] - atom1z;*/
                    float dx = atom1x - coord[3*kk+0];
                    float dy = atom1y - coord[3*kk+1];
                    float dz = atom1z - coord[3*kk+2];
                    float r = sqrt(dx*dx+dy*dy+dz*dz);
                    float qr = q_pt * r; 
                    float sqr = sin(qr) / qr;
                    float dsqr = cos(qr) - sqr;
                    float prefac = FF_kj * dsqr / r / r;
                    prefac += prefac;
                    S_calccs += FF_kj * sqr;
                    f_ptxcs += prefac * dx;
                    f_ptycs += prefac * dy;
                    f_ptzcs += prefac * dz;
                }
            }
            S_calcc[ii*num_atom2+jj] = S_calccs;
            f_ptxc[ii*num_atom2+jj] = f_ptxcs;
            f_ptyc[ii*num_atom2+jj] = f_ptycs;
            f_ptzc[ii*num_atom2+jj] = f_ptzcs;
        }
        
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        S_calc[ii] = (double) S_calcc[ii * num_atom2];
        __syncthreads();
        
        if (threadIdx.x == 0) {
            
            // Here comes in the past scat
            // Scat is calced to (S_new + ((N-1) / N) S_old) / N-1
            // Remember to convert S_new to double or set an array for it.
            S_calc[ii] += S_old[ii] * (EMA_norm - 1.0);
            S_calc[ii] /= EMA_norm;
            // Update old scattering
            S_old[ii] = S_calc[ii];
            
            Aq[ii] = (float)S_calc[ii] - q_S_ref_dS[ii+num_q];
            Aq[ii] *= -alpha;
            Aq[ii] += q_S_ref_dS[ii + 2*num_q];
            Aq[ii] *= k_chi / sigma2_pt;
            Aq[ii] += Aq[ii];
        }
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
        }
    }
}

__global__ void __launch_bounds__(1024,2) scat_calc_EMA_surf_grad (
//__global__ void scat_calc_EMA_surf_grad (
    float *coord, 
    int *Ele,
    float *q_S_ref_dS, 
    double *S_calc, 
    int num_atom,   
    int num_q,     
    int num_ele,   
    float *Aq, 
    float alpha,   
    float k_chi,    
    float *sigma2,  
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    float *S_calcc, 
    int num_atom2,
    float *surf_grad,
    float *FF_full,
    double *S_old,
    double EMA_norm) {

    // EMA_norm is computed on the host. See Chen & Hub, Biophysics 2015 2573-2584.

    float q_pt, sigma2_pt; 

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q_S_ref_dS[ii];
        sigma2_pt = sigma2[ii];
        // Calculate scattering for Aq
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            // for every atom jj
            float atom1x = coord[3*jj+0];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            // For surface gradient
            float S_calccs = 0.0;
            float f_ptxcs = 0.0;
            float f_ptycs = 0.0;
            float f_ptzcs = 0.0;
            float FF_j = FF_full[ii * num_atom2 + jj];
            for (int kk = 0; kk < num_atom; kk++) {
                float FF_kj = FF_j * FF_full[ii * num_atom2 + kk];
                /*if (kk == jj) {
                    S_calccs += FF_kj;
                } else if (q_pt == 0.0) {
                    float dx = coord[3*kk+0] - atom1x;
                    float dy = coord[3*kk+1] - atom1y;
                    float dz = coord[3*kk+2] - atom1z;
                    float r = sqrt(dx*dx+dy*dy+dz*dz);
                    float prefac = surf_grad[3*jj+0] * dx; 
                    prefac += surf_grad[3*jj+1] * dy;
                    prefac += surf_grad[3*jj+2] * dz;
                    prefac *= FF_j / r / r;
                    prefac += prefac;
                    S_calccs += FF_kj;
                    f_ptxcs += prefac * dx;
                    f_ptycs += prefac * dy;
                    f_ptzcs += prefac * dz;
                } else {
                    float dx = coord[3*kk+0] - atom1x;
                    float dy = coord[3*kk+1] - atom1y;
                    float dz = coord[3*kk+2] - atom1z;
                    float r = sqrt(dx*dx+dy*dy+dz*dz);
                    float qr = q_pt * r; 
                    float sqr = sin(qr) / qr;
                    float dsqr = cos(qr) - sqr;
                    float prefac = surf_grad[3*jj+0] * dx; 
                    prefac += surf_grad[3*jj+1] * dy;
                    prefac += surf_grad[3*jj+2] * dz;
                    prefac *= FF_j * sqr; 
                    prefac += FF_kj * dsqr / r / r;
                    prefac += prefac;
                    S_calccs += FF_kj * sqr;
                    f_ptxcs += prefac * dx;
                    f_ptycs += prefac * dy;
                    f_ptzcs += prefac * dz;
                }*/
                // for every atom kk
                float dx = coord[3*kk+0] - atom1x;
                float dy = coord[3*kk+1] - atom1y;
                float dz = coord[3*kk+2] - atom1z;
                /*float dx = atom1x - coord[3*kk+0];
                float dy = atom1y - coord[3*kk+1];
                float dz = atom1z - coord[3*kk+2];*/
                float r = sqrt(dx*dx+dy*dy+dz*dz);
                float prefac = surf_grad[3*jj+0] * dx; 
                prefac += surf_grad[3*jj+1] * dy;
                prefac += surf_grad[3*jj+2] * dz;
                prefac *= FF_j; 
                float sqr = 1.0;
                if (kk == jj) r = 1.0;
                if (kk == jj || q_pt == 0.0) {
                    sqr = 1.0;
                } else {
                    float qr = q_pt * r; 
                    sqr = sin(qr) / qr;
                    float dsqr = cos(qr) - sqr;
                    prefac *= sqr;
                    prefac += FF_kj * dsqr;
                }
                prefac += prefac;
                prefac = prefac / r / r;
                //printf("sqr = %6.3f, FF_kj = %6.3f, prod = %6.3f \n", sqr, FF_kj, sqr*FF_kj);
                S_calccs += FF_kj * sqr;
                f_ptxcs += prefac * dx;
                f_ptycs += prefac * dy;
                f_ptzcs += prefac * dz;
            }
            //printf("S_calccs(q=%.3f) = %6.3f\n", q_pt, S_calccs);
            S_calcc[ii*num_atom2+jj] = S_calccs;
            f_ptxc[ii*num_atom2+jj] = f_ptxcs;
            f_ptyc[ii*num_atom2+jj] = f_ptycs;
            f_ptzc[ii*num_atom2+jj] = f_ptzcs;
        }
        
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        S_calc[ii] = (double) S_calcc[ii * num_atom2];
        __syncthreads();
        
        if (threadIdx.x == 0) {
            
            // Here comes in the past scat
            // Scat is calced to (S_new + ((N-1) / N) S_old) / N-1
            // Remember to convert S_new to double or set an array for it.
            S_calc[ii] += S_old[ii] * (EMA_norm - 1.0);
            S_calc[ii] /= EMA_norm;
            // Update old scattering
            S_old[ii] = S_calc[ii];
            
            Aq[ii] = (float)S_calc[ii] - q_S_ref_dS[ii+num_q];
            Aq[ii] *= -alpha;
            Aq[ii] += q_S_ref_dS[ii + 2*num_q];
            Aq[ii] *= k_chi / sigma2_pt;
            Aq[ii] += Aq[ii];
        }
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
        }
    }
}

__global__ void __launch_bounds__(1024,2) scat_calc_bin (
    float *coord, 
    int *Ele,
    float *q_S_ref_dS, 
    float *S_calc, 
    int num_atom,   
    int num_q,     
    int num_ele,   
    float *Aq, 
    float alpha,   
    float k_chi,    
    float *sigma2,
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    float *S_calcc, 
    int num_atom2,
    float *FF_full
    /*float *q_a_r,
    float *q_a_rx,
    float *q_a_ry,
    float *q_a_rz*/) { 

    // q_a_r is a 3D matrix of dimension num_q, 1024, 401
    // every q will use a slice of that matrix, and every atom jj % 1024 will use an array
    // of the slice, recording the FFT amplitude.

    float q_pt; 
    //__shared__ float sqr[256];  // This is binned sin(q * r) / (q * r)
    //__shared__ float csqrr[256]; // This is binned (cos(q * r) - sin(q * r) / (q * r)) / r^2
    __shared__ int q_a_r [1024];
    __shared__ int q_a_rx[1024];
    __shared__ int q_a_ry[1024];
    __shared__ int q_a_rz[1024];
    /*__shared__ float q_a_r2 [1024];
    __shared__ float q_a_rx2[1024];
    __shared__ float q_a_ry2[1024];
    __shared__ float q_a_rz2[1024];*/
 
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q_S_ref_dS[ii];
        for (int kk = threadIdx.x; kk < 1024; kk += blockDim.x) {
            q_a_r [kk] = 0;
            q_a_rx[kk] = 0;
            q_a_ry[kk] = 0;
            q_a_rz[kk] = 0;
        }
        // Determine the sqr and csqrr
        /*for (int jj = threadIdx.x; jj < 256; jj += blockDim.x) {
            float r = (float)jj * 0.5 + 0.25;
            float qr = q_pt * r;
            sqr[jj] = sin(qr) / qr;
            float dsqr = cos(qr) - sqr[jj];
            csqrr[jj] =  dsqr / r / r;
        }*/
        //__syncthreads();
        // Calculate scattering for Aq
        for (int jj = 0; jj < num_atom; jj ++) {
            /*for (int kk = threadIdx.x; kk < 1024; kk += blockDim.x) {
                q_a_r [kk] = 0;
                q_a_rx[kk] = 0;
                q_a_ry[kk] = 0;
                q_a_rz[kk] = 0;*/
                /*q_a_r2 [kk] = 0.0;
                q_a_rx2[kk] = 0.0;
                q_a_ry2[kk] = 0.0;
                q_a_rz2[kk] = 0.0;*/
            //}

            //__syncthreads(); // This one is essential
            // for every atom jj
            float atom1x = coord[3*jj+0];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            float atom1FF = FF_full[ii*num_atom2 +jj];
            //__syncthreads(); // This one is essential
            for (int kk = threadIdx.x; kk < num_atom; kk+= blockDim.x) {
                // for every atom kk
                //float FF_kj = atom1FF * FF_full[ii *num_atom2 + kk];
                float FF_kj = FF_full[ii *num_atom2 + kk];
                //if (q_pt == 0.0 || kk == jj) {
                //    S_calccs += FF_kj;
                //} else {
                float dx = coord[3*kk+0] - atom1x;
                float dy = coord[3*kk+1] - atom1y;
                float dz = coord[3*kk+2] - atom1z;
                float r = sqrt(dx*dx+dy*dy+dz*dz);
                //if (r < 64) {
                    int idz = r += r; // Because we're using 0.5 A bins.
                    /*int ida = threadIdx.x / 256 * 256;
                    atomicAdd(&q_a_r [idz+ida], (int)(FF_kj * 1e6));
                    atomicAdd(&q_a_rx[idz+ida], (int)(2e6 * FF_kj * dx));
                    atomicAdd(&q_a_ry[idz+ida], (int)(2e6 * FF_kj * dy));
                    atomicAdd(&q_a_rz[idz+ida], (int)(2e6 * FF_kj * dz));*/
                    int dumm = atom1FF * FF_kj * 1e4;
                    atomicAdd(&q_a_r [idz], dumm);
                    dumm = atom1FF * FF_kj * dx * 2e4;
                    atomicAdd(&q_a_rx[idz], dumm);
                    dumm = atom1FF * FF_kj * dy * 2e4;
                    atomicAdd(&q_a_ry[idz], dumm);
                    dumm = atom1FF * FF_kj * dz * 2e4;
                    atomicAdd(&q_a_rz[idz], dumm);

                    /*atomicAdd(&q_a_r [idz], (int)(FF_kj * 1e4));
                    atomicAdd(&q_a_rx[idz], (int)(2e4 * FF_kj * dx));
                    atomicAdd(&q_a_ry[idz], (int)(2e4 * FF_kj * dy));
                    atomicAdd(&q_a_rz[idz], (int)(2e4 * FF_kj * dz));*/
                        /*float qr = q_pt * r; 
                        float sqr = sin(qr) / qr;
                        float dsqr = cos(qr) - sqr;
                        float prefac = FF_kj * dsqr / r / r;
                        prefac += prefac;
                        S_calccs += FF_kj * sqr;
                        f_ptxcs += prefac * dx;
                        f_ptycs += prefac * dy;
                        f_ptzcs += prefac * dz;*/
                    //}
                //}
            }

            /*for (int stride = 512; stride > 128; stride >>= 1) {
                __syncthreads();
                for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                    q_a_r[iAccum] += q_a_r[iAccum + stride];
                    q_a_rx[iAccum] += q_a_rx[iAccum + stride];
                    q_a_ry[iAccum] += q_a_ry[iAccum + stride];
                    q_a_rz[iAccum] += q_a_rz[iAccum + stride];
                }
            }*/
            __syncthreads();

            for (int kk = threadIdx.x; kk < 256; kk += blockDim.x) {
                float r = (float)kk * 0.5 + 0.25;
                float qr = q_pt * r;
                float sqr = sin(qr) / qr;
                float dsqr = cos(qr) - sqr;
                float csqrr = dsqr / r / r;
                //atomicAdd(&S_calcc[ii*num_atom2+jj],(float)( 1e-6 * q_a_r[kk]  *   sqr[kk]));
                //atomicAdd(&f_ptxc[ii*num_atom2+jj], (float)( 1e-6 * q_a_rx[kk] * csqrr[kk]));
                //atomicAdd(&f_ptyc[ii*num_atom2+jj], (float)( 1e-6 * q_a_ry[kk] * csqrr[kk]));
                //atomicAdd(&f_ptzc[ii*num_atom2+jj], (float)( 1e-6 * q_a_rz[kk] * csqrr[kk]));
                /*atomicAdd(&S_calcc[ii*num_atom2+jj],(float)( 1e-6 * q_a_r[kk]  *   sqr));
                atomicAdd(&f_ptxc[ii*num_atom2+jj], (float)( 1e-6 * q_a_rx[kk] * csqrr));
                atomicAdd(&f_ptyc[ii*num_atom2+jj], (float)( 1e-6 * q_a_ry[kk] * csqrr));
                atomicAdd(&f_ptzc[ii*num_atom2+jj], (float)( 1e-6 * q_a_rz[kk] * csqrr));*/
                /*q_a_r2[kk] = q_a_r[kk] * sqr;
                q_a_rx2[kk] = q_a_rx[kk] * csqrr;
                q_a_ry2[kk] = q_a_ry[kk] * csqrr;
                q_a_rz2[kk] = q_a_rz[kk] * csqrr;*/
                atomicAdd(&S_calcc[ii*num_atom2+jj],q_a_r[kk]  *   sqr);
                q_a_r[kk] = 0;
                atomicAdd(&f_ptxc[ii*num_atom2+jj], q_a_rx[kk] * csqrr);
                q_a_rx[kk] = 0;
                atomicAdd(&f_ptyc[ii*num_atom2+jj], q_a_ry[kk] * csqrr);
                q_a_ry[kk] = 0;
                atomicAdd(&f_ptzc[ii*num_atom2+jj], q_a_rz[kk] * csqrr);
                q_a_rz[kk] = 0;
            }
            __syncthreads();

            /*
            for (int stride = 1024 / 2; stride > 0; stride >>= 1) {
                __syncthreads();
                for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                    q_a_r2[iAccum] += q_a_r2[stride + iAccum];
                    q_a_rx2[iAccum] += q_a_rx2[stride + iAccum];
                    q_a_ry2[iAccum] += q_a_ry2[stride + iAccum];
                    q_a_rz2[iAccum] += q_a_rz2[stride + iAccum];
                }
            }
            //__syncthreads();
            if (threadIdx.x == 0) {
                S_calcc[ii*num_atom2+jj]+= q_a_r2[0] ;
                f_ptxc [ii*num_atom2+jj]+= q_a_rx2[0];
                f_ptyc [ii*num_atom2+jj]+= q_a_ry2[0];
                f_ptzc [ii*num_atom2+jj]+= q_a_rz2[0];
            }
            */
 
        }
        
        //if (ii == 0 && threadIdx.x == 0) printf("\nS_calcc[0] = %.3f \n",S_calcc[0]);

        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        //if (threadIdx.x == 0) {
            S_calc[ii] = S_calcc[ii * num_atom2]*1e-4;
            Aq[ii] = S_calc[ii] - q_S_ref_dS[ii+num_q];
            Aq[ii] *= -alpha;
            Aq[ii] += q_S_ref_dS[ii + 2*num_q];
            Aq[ii] *= k_chi / sigma2[ii];
            Aq[ii] += Aq[ii];
        //}
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * alpha * 1e-4;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * alpha * 1e-4;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * alpha * 1e-4;
        }
    }
}
__global__ void __launch_bounds__(1024,2) scat_calc_bin_unroll (
    float *coord, 
    int *Ele,
    float *q_S_ref_dS, 
    float *S_calc, 
    int num_atom,   
    int num_q,     
    int num_ele,   
    float *Aq, 
    float alpha,   
    float k_chi,    
    float *sigma2,
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    float *S_calcc, 
    int num_atom2,
    float *FF_full   
    ) { 

    float q_pt; 
    /*__shared__ int q_a_r [4][511];
    __shared__ int q_a_rx[4][511];
    __shared__ int q_a_ry[4][511];
    __shared__ int q_a_rz[4][511];*/
    /*__shared__ int q_a_r [1024];
    __shared__ int q_a_rx[1024];
    __shared__ int q_a_ry[1024];
    __shared__ int q_a_rz[1024];
    __shared__ int q_a_r2 [1024];
    __shared__ int q_a_rx2[1024];
    __shared__ int q_a_ry2[1024];
    __shared__ int q_a_rz2[1024];*/
    __shared__ int q_a_r [512];
    __shared__ int q_a_rx[512];
    __shared__ int q_a_ry[512];
    __shared__ int q_a_rz[512];
    __shared__ int q_a_r2 [512];
    __shared__ int q_a_rx2[512];
    __shared__ int q_a_ry2[512];
    __shared__ int q_a_rz2[512];
    /*__shared__ float atomx[4];
    __shared__ float atomy[4];
    __shared__ float atomz[4];
    __shared__ float atomFF[4];*/
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q_S_ref_dS[ii];
        for (int kk = threadIdx.x; kk < 256; kk += blockDim.x) {
            /*for (int ll = 0; ll < 4; ll++) {
                q_a_r[ll][kk] = 0;
                q_a_rx[ll][kk] = 0;
                q_a_ry[ll][kk] = 0;
                q_a_rz[ll][kk] = 0;
            }*/
            q_a_r [kk] = 0;
            q_a_rx[kk] = 0;
            q_a_ry[kk] = 0;
            q_a_rz[kk] = 0;
            q_a_r2 [kk] = 0;
            q_a_rx2[kk] = 0;
            q_a_ry2[kk] = 0;
            q_a_rz2[kk] = 0;
        }
        for (int jj = 0; jj < num_atom; jj += 2) {
            // for every atom jj
            /*for (int kk = 0; kk < 4; kk++) {
                atomx[kk] = coord[3*jj+3*kk+0];
                atomy[kk] = coord[3*jj+3*kk+1];
                atomz[kk] = coord[3*jj+3*kk+2];
                atomFF[kk] = FF_full[ii*num_atom2 + jj + 3 * kk];
            }*/
            float atom1x = coord[3*jj+0];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            float atom2x = coord[3*jj+3];
            float atom2y = coord[3*jj+4];
            float atom2z = coord[3*jj+5];
            float atom1FF = FF_full[ii*num_atom2 +jj];
            float atom2FF = FF_full[ii*num_atom2 + jj + 1];
            for (int kk = threadIdx.x; kk < num_atom; kk+= blockDim.x) {
                // for every atom kk
                float FF_kj = FF_full[ii *num_atom2 + kk];
                /*for (int ll = 0; ll < 4; ll++) {

                    float dx = coord[3*kk+0] - atomx[ll];
                    float dy = coord[3*kk+1] - atomy[ll];
                    float dz = coord[3*kk+2] - atomz[ll];
                    float r = sqrt(dx*dx+dy*dy+dz*dz);
                    int idz = r += r; // Because we're using 0.5 A bins.
                    int dumm = atomFF[ll] * FF_kj * 1e4;
                    atomicAdd(&q_a_r [ll][idz], dumm);
                    dumm = atomFF[ll] * FF_kj * dx * 2e4;
                    atomicAdd(&q_a_rx[ll][idz], dumm);
                    dumm = atomFF[ll] * FF_kj * dy * 2e4;
                    atomicAdd(&q_a_ry[ll][idz], dumm);
                    dumm = atomFF[ll] * FF_kj * dz * 2e4;
                    atomicAdd(&q_a_rz[ll][idz], dumm);

                }*/
                float x = coord[3*kk+0];
                float y = coord[3*kk+1];
                float z = coord[3*kk+2];
                float dx = x - atom1x;
                float dy = y - atom1y;
                float dz = z - atom1z;
                /*float dx = coord[3*kk+0] - atom1x;
                float dy = coord[3*kk+1] - atom1y;
                float dz = coord[3*kk+2] - atom1z;*/
                float r = sqrt(dx*dx+dy*dy+dz*dz);
                int idz = r += r; // Because we're using 0.5 A bins.
                //if (idz < 256) {
                int dumm = atom1FF * FF_kj * 10000;
                atomicAdd(&q_a_r [idz], dumm);
                dumm = atom1FF * FF_kj * dx * 20000;
                atomicAdd(&q_a_rx[idz], dumm);
                dumm = atom1FF * FF_kj * dy * 20000;
                atomicAdd(&q_a_ry[idz], dumm);
                dumm = atom1FF * FF_kj * dz * 20000;
                atomicAdd(&q_a_rz[idz], dumm);
/*
                // Try to avoid double mul
                int dumm = atom1FF * 1e4;
                dumm *= FF_kj;
                atomicAdd(&q_a_r [idz], dumm);
                dumm = dx * 2e4;
                dumm *= atom1FF * FF_kj;
                atomicAdd(&q_a_rx[idz], dumm);
                dumm = dy * 2e4;
                dumm *= atom1FF * FF_kj;
                atomicAdd(&q_a_ry[idz], dumm);
                dumm = dz * 2e4;
                dumm *= atom1FF * FF_kj;
                atomicAdd(&q_a_rz[idz], dumm);
*/
                //}
                /*dx = coord[3*kk+0] - atom2x;
                dy = coord[3*kk+1] - atom2y;
                dz = coord[3*kk+2] - atom2z;*/
                dx = x - atom2x;
                dy = y - atom2y;
                dz = z - atom2z;
                r = sqrt(dx*dx+dy*dy+dz*dz);
                idz = r += r; // Because we're using 0.5 A bins.
                //if (idz < 256) {
                //int dumm = atom2FF * FF_kj * 1e4;
                dumm = atom2FF * FF_kj * 10000;
                atomicAdd(&q_a_r2 [idz], dumm);
                dumm = atom2FF * FF_kj * dx * 20000;
                atomicAdd(&q_a_rx2[idz], dumm);
                dumm = atom2FF * FF_kj * dy * 20000;
                atomicAdd(&q_a_ry2[idz], dumm);
                dumm = atom2FF * FF_kj * dz * 20000;
                atomicAdd(&q_a_rz2[idz], dumm);

                //}
/*                dumm = atom1FF * 1e4;
                dumm *= FF_kj;
                atomicAdd(&q_a_r2 [idz], dumm);
                dumm = dx * 2e4;
                dumm *= atom1FF * FF_kj;
                atomicAdd(&q_a_rx2[idz], dumm);
                dumm = dy * 2e4;
                dumm *= atom1FF * FF_kj;
                atomicAdd(&q_a_ry2[idz], dumm);
                dumm = dz * 2e4;
                dumm *= atom1FF * FF_kj;
                atomicAdd(&q_a_rz2[idz], dumm);*/
            }

            __syncthreads();

            for (int kk = threadIdx.x; kk < 256; kk += blockDim.x) {
                float r = (float)kk * 0.5 + 0.25;
                float qr = q_pt * r;
                float sqr = sin(qr) / qr;
                float dsqr = cos(qr) - sqr;
                float csqrr = dsqr / r / r;
                /*for (int ll = 0; ll < 4; ll++) {
                    atomicAdd(&S_calcc[ii*num_atom2+jj],q_a_r[ll][kk]  *   sqr);
                    q_a_r[ll][kk] = 0;
                    atomicAdd(&f_ptxc[ii*num_atom2+jj],q_a_rx[ll][kk]  *   csqrr);
                    q_a_rx[ll][kk] = 0;
                    atomicAdd(&f_ptyc[ii*num_atom2+jj],q_a_ry[ll][kk]  *   csqrr);
                    q_a_ry[ll][kk] = 0;
                    atomicAdd(&f_ptzc[ii*num_atom2+jj],q_a_rz[ll][kk]  *   csqrr);
                    q_a_rz[ll][kk] = 0;
                }*/
                atomicAdd(&S_calcc[ii*num_atom2+jj],q_a_r[kk]  *   sqr);
                atomicAdd(&S_calcc[ii*num_atom2+jj+1],q_a_r2[kk]  *   sqr);
                q_a_r[kk] = 0;
                q_a_r2[kk] = 0;
                atomicAdd(&f_ptxc[ii*num_atom2+jj], q_a_rx[kk] * csqrr);
                atomicAdd(&f_ptxc[ii*num_atom2+jj+1], q_a_rx2[kk] * csqrr);
                q_a_rx[kk] = 0;
                q_a_rx2[kk] = 0;
                atomicAdd(&f_ptyc[ii*num_atom2+jj], q_a_ry[kk] * csqrr);
                atomicAdd(&f_ptyc[ii*num_atom2+jj+1], q_a_ry2[kk] * csqrr);
                q_a_ry[kk] = 0;
                q_a_ry2[kk] = 0;
                atomicAdd(&f_ptzc[ii*num_atom2+jj], q_a_rz[kk] * csqrr);
                atomicAdd(&f_ptzc[ii*num_atom2+jj+1], q_a_rz2[kk] * csqrr);
                q_a_rz[kk] = 0;
                q_a_rz2[kk] = 0;
            }
            __syncthreads();

            /*
            for (int stride = 1024 / 2; stride > 0; stride >>= 1) {
                __syncthreads();
                for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                    q_a_r2[iAccum] += q_a_r2[stride + iAccum];
                    q_a_rx2[iAccum] += q_a_rx2[stride + iAccum];
                    q_a_ry2[iAccum] += q_a_ry2[stride + iAccum];
                    q_a_rz2[iAccum] += q_a_rz2[stride + iAccum];
                }
            }
            //__syncthreads();
            if (threadIdx.x == 0) {
                S_calcc[ii*num_atom2+jj]+= q_a_r2[0] ;
                f_ptxc [ii*num_atom2+jj]+= q_a_rx2[0];
                f_ptyc [ii*num_atom2+jj]+= q_a_ry2[0];
                f_ptzc [ii*num_atom2+jj]+= q_a_rz2[0];
            }
            */
 
        }
        
        //if (ii == 0 && threadIdx.x == 0) printf("\nS_calcc[0] = %.3f \n",S_calcc[0]);

        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        //if (threadIdx.x == 0) {
            S_calc[ii] = S_calcc[ii * num_atom2]*1e-4;
            Aq[ii] = S_calc[ii] - q_S_ref_dS[ii+num_q];
            Aq[ii] *= -alpha;
            Aq[ii] += q_S_ref_dS[ii + 2*num_q];
            Aq[ii] *= k_chi / sigma2[ii];
            Aq[ii] += Aq[ii];
        //}
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * alpha * 1e-4;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * alpha * 1e-4;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * alpha * 1e-4;
        }
    }
}
__global__ void __launch_bounds__(1024,2) scat_calc_bin_unroll2 (
    float *coord, 
    int *Ele,
    float *q_S_ref_dS, 
    float *S_calc, 
    int num_atom,   
    int num_q,     
    int num_ele,   
    float *Aq, 
    float alpha,   
    float k_chi,    
    float *sigma2,
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    float *S_calcc, 
    int num_atom2,
    float *FF_full   
    ) { 

    float q_pt; 
    __shared__ int q_a_r [2048];
    __shared__ int q_a_rx[2048];
    __shared__ int q_a_ry[2048];
    __shared__ int q_a_rz[2048];
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q_S_ref_dS[ii];
        for (int kk = threadIdx.x; kk < 2048; kk += blockDim.x) {
            q_a_r[kk] = 0;
            q_a_rx[kk] = 0;
            q_a_ry[kk] = 0;
            q_a_rz[kk] = 0;
        }
        for (int jj = 0; jj < num_atom; jj += 3) {
            // for every atom jj
            /*for (int kk = 0; kk < 4; kk++) {
                atomx[kk] = coord[3*jj+3*kk+0];
                atomy[kk] = coord[3*jj+3*kk+1];
                atomz[kk] = coord[3*jj+3*kk+2];
                atomFF[kk] = FF_full[ii*num_atom2 + jj + 3 * kk];
            }*/
            float atom1x = coord[3*jj+0];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            float atom2x = coord[3*jj+3];
            float atom2y = coord[3*jj+4];
            float atom2z = coord[3*jj+5];
            float atom3x = coord[3*jj+6];
            float atom3y = coord[3*jj+7];
            float atom3z = coord[3*jj+8];
            //float atom4x = coord[3*jj+9];
            //float atom4y = coord[3*jj+10];
            //float atom4z = coord[3*jj+11];
            float atom1FF = FF_full[ii*num_atom2 + jj];
            float atom2FF = FF_full[ii*num_atom2 + jj + 1];
            float atom3FF = FF_full[ii*num_atom2 + jj + 2];
            //float atom4FF = FF_full[ii*num_atom2 + jj + 3];
            for (int kk = threadIdx.x; kk < num_atom; kk+= blockDim.x) {
                // for every atom kk
                float FF_kj = FF_full[ii *num_atom2 + kk];
                float x = coord[3*kk+0];
                float y = coord[3*kk+1];
                float z = coord[3*kk+2];
                float dx = x - atom1x;
                float dy = y - atom1y;
                float dz = z - atom1z;
                float r = sqrt(dx*dx+dy*dy+dz*dz);
                int idz = r += r; // Because we're using 0.5 A bins.
                int dumm = atom1FF * FF_kj * 10000;
                atomicAdd(&q_a_r [idz], dumm);
                dumm = atom1FF * FF_kj * dx * 20000;
                atomicAdd(&q_a_rx[idz], dumm);
                dumm = atom1FF * FF_kj * dy * 20000;
                atomicAdd(&q_a_ry[idz], dumm);
                dumm = atom1FF * FF_kj * dz * 20000;
                atomicAdd(&q_a_rz[idz], dumm);

                dx = x - atom2x;
                dy = y - atom2y;
                dz = z - atom2z;
                r = sqrt(dx*dx+dy*dy+dz*dz);
                idz = r += r; // Because we're using 0.5 A bins.
                dumm = atom2FF * FF_kj * 10000;
                atomicAdd(&q_a_r [idz+512], dumm);
                dumm = atom2FF * FF_kj * dx * 20000;
                atomicAdd(&q_a_rx[idz+512], dumm);
                dumm = atom2FF * FF_kj * dy * 20000;
                atomicAdd(&q_a_ry[idz+512], dumm);
                dumm = atom2FF * FF_kj * dz * 20000;
                atomicAdd(&q_a_rz[idz+512], dumm);

                dx = x - atom3x;
                dy = y - atom3y;
                dz = z - atom3z;
                r = sqrt(dx*dx+dy*dy+dz*dz);
                idz = r += r; // Because we're using 0.5 A bins.
                dumm = atom3FF * FF_kj * 10000;
                atomicAdd(&q_a_r [idz+1024], dumm);
                dumm = atom3FF * FF_kj * dx * 20000;
                atomicAdd(&q_a_rx[idz+1024], dumm);
                dumm = atom3FF * FF_kj * dy * 20000;
                atomicAdd(&q_a_ry[idz+1024], dumm);
                dumm = atom3FF * FF_kj * dz * 20000;
                atomicAdd(&q_a_rz[idz+1024], dumm);

/*                dx = x - atom4x;
                dy = y - atom4y;
                dz = z - atom4z;
                r = sqrt(dx*dx+dy*dy+dz*dz);
                idz = r += r; // Because we're using 0.5 A bins.
                dumm = atom4FF * FF_kj * 10000;
                atomicAdd(&q_a_r [idz+1536], dumm);
                dumm = atom4FF * FF_kj * dx * 20000;
                atomicAdd(&q_a_rx[idz+1536], dumm);
                dumm = atom4FF * FF_kj * dy * 20000;
                atomicAdd(&q_a_ry[idz+1536], dumm);
                dumm = atom4FF * FF_kj * dz * 20000;
                atomicAdd(&q_a_rz[idz+1536], dumm);
*/
            }

            __syncthreads();

            for (int kk = threadIdx.x; kk < 1536; kk += blockDim.x) {
                float r = (float)(kk % 512 * 0.5 + 0.25);
                float qr = q_pt * r;
                float sqr = sin(qr) / qr;
                float dsqr = cos(qr) - sqr;
                float csqrr = dsqr / r / r;
                    atomicAdd(&S_calcc[ii*num_atom2+jj],q_a_r[kk]  *   sqr);
                    q_a_r[kk] = 0;
                    atomicAdd(&f_ptxc[ii*num_atom2+jj],q_a_rx[kk]  *   csqrr);
                    q_a_rx[kk] = 0;
                    atomicAdd(&f_ptyc[ii*num_atom2+jj],q_a_ry[kk]  *   csqrr);
                    q_a_ry[kk] = 0;
                    atomicAdd(&f_ptzc[ii*num_atom2+jj],q_a_rz[kk]  *   csqrr);
                    q_a_rz[kk] = 0;
            }
            __syncthreads();

            /*
            for (int stride = 1024 / 2; stride > 0; stride >>= 1) {
                __syncthreads();
                for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                    q_a_r2[iAccum] += q_a_r2[stride + iAccum];
                    q_a_rx2[iAccum] += q_a_rx2[stride + iAccum];
                    q_a_ry2[iAccum] += q_a_ry2[stride + iAccum];
                    q_a_rz2[iAccum] += q_a_rz2[stride + iAccum];
                }
            }
            //__syncthreads();
            if (threadIdx.x == 0) {
                S_calcc[ii*num_atom2+jj]+= q_a_r2[0] ;
                f_ptxc [ii*num_atom2+jj]+= q_a_rx2[0];
                f_ptyc [ii*num_atom2+jj]+= q_a_ry2[0];
                f_ptzc [ii*num_atom2+jj]+= q_a_rz2[0];
            }
            */
 
        }
        
        //if (ii == 0 && threadIdx.x == 0) printf("\nS_calcc[0] = %.3f \n",S_calcc[0]);

        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        //if (threadIdx.x == 0) {
            S_calc[ii] = S_calcc[ii * num_atom2]*1e-4;
            Aq[ii] = S_calc[ii] - q_S_ref_dS[ii+num_q];
            Aq[ii] *= -alpha;
            Aq[ii] += q_S_ref_dS[ii + 2*num_q];
            Aq[ii] *= k_chi / sigma2[ii];
            Aq[ii] += Aq[ii];
        //}
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * alpha * 1e-4;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * alpha * 1e-4;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * alpha * 1e-4;
        }
    }
}
/*
__global__ void FF_calc_T (
    float *FF_full,
    float *FF_full_T,
    int num_q2, 
    int num_atom2) {


    int TILE_DIM = 32;
    int BLOCK_ROWS = num_atom2 * num_q2 / TILE_DIM;
    __shared__ float tile[TILE_DIM * TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = FF_full[(y+j)*width + x];

    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        FF_full_T[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];  

}
*/

__global__ void __launch_bounds__(1024,2) scat_calc_bin_T (
    float *coord, 
    int *Ele,
    float *q_S_ref_dS, 
    float *S_calc, 
    int num_atom,   
    int num_q,    
    int num_q2, 
    int num_ele,   
    float *Aq, 
    float alpha,   
    float k_chi,    
    float sigma2,
    float *f_ptxc_T, 
    float *f_ptyc_T, 
    float *f_ptzc_T, 
    float *S_calcc_T, 
    int num_atom2,
    float *FF_full_T,
    float *a_r_q,
    float *a_r_qx,
    float *a_r_qy,
    float *a_r_qz) { 

    // a_r_q is a 3D matrix of dimension 896 (16*56 for now), 416, num_q2
    // every atom % 1024 will use a slice of that matrix, and every q vector will use an array
    // of the slice, recording the FFT amplitude.

    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        // Flush a_r_q

        for (int jj = threadIdx.x; jj < num_q; jj += blockDim.x) {
            for (int kk = 0; kk < 416; kk ++) {
                a_r_q [ii % 896 * num_q2 * 416 + kk * num_q2 + jj] = 0.0;
                a_r_qx[ii % 896 * num_q2 * 416 + kk * num_q2 + jj] = 0.0;
                a_r_qy[ii % 896 * num_q2 * 416 + kk * num_q2 + jj] = 0.0;
                a_r_qz[ii % 896 * num_q2 * 416 + kk * num_q2 + jj] = 0.0;
            }


        }
 
        float atom1x = coord[ii+0];
        float atom1y = coord[ii+1*num_atom];
        float atom1z = coord[ii+2*num_atom];
        __syncthreads();
        // Calculate scattering for Aq
        for (int jj = threadIdx.x; jj < num_q; jj += blockDim.x) {
            float atom1FF = FF_full_T[ii * num_q2 + jj];
            //if (ii == 0 && jj == 0) printf("\natom1FF is %.3f\n", atom1FF);
            float q_pt = q_S_ref_dS[jj];
            // for every q jj
            for (int kk = 0; kk < num_atom; kk++) {
                // for every atom kk
                float FF_kj = atom1FF * FF_full_T[kk * num_q2 + jj];
                float dx = coord[kk+0] - atom1x;
                float dy = coord[kk+1*num_atom] - atom1y;
                float dz = coord[kk+2*num_atom] - atom1z;
                float r = sqrt(dx*dx+dy*dy+dz*dz);
                /*if (ii == 0 && jj == 0 && kk == 1) {
                    printf("\natom1 = %8.3f, %8.3f, %8.3f \natom2 = %8.3f, %8.3f, %.3f, \nr01 = %.3f\n",atom1x, atom1y, atom1z, coord[kk], coord[kk+num_atom], coord[kk+2*num_atom], r);
                }*/
                if (r < 200.0) {
                    int idz = r * 2; // Because we're using 0.5 A bins.
                    /*int ida = threadIdx.x / 256 * 256;
                    atomicAdd(&q_a_r [idz+ida], (int)(FF_kj * 1e6));
                    atomicAdd(&q_a_rx[idz+ida], (int)(2e6 * FF_kj * dx));
                    atomicAdd(&q_a_ry[idz+ida], (int)(2e6 * FF_kj * dy));
                    atomicAdd(&q_a_rz[idz+ida], (int)(2e6 * FF_kj * dz));*/
                    a_r_q [ii % 896*num_q2*416+idz*num_q2+jj] += FF_kj;
                    a_r_qx[ii % 896*num_q2*416+idz*num_q2+jj] += FF_kj * dx;
                    a_r_qx[ii % 896*num_q2*416+idz*num_q2+jj] += FF_kj * dx;
                    a_r_qy[ii % 896*num_q2*416+idz*num_q2+jj] += FF_kj * dy;
                    a_r_qy[ii % 896*num_q2*416+idz*num_q2+jj] += FF_kj * dy;
                    a_r_qz[ii % 896*num_q2*416+idz*num_q2+jj] += FF_kj * dz;
                    a_r_qz[ii % 896*num_q2*416+idz*num_q2+jj] += FF_kj * dz;
    
                        /*float qr = q_pt * r; 
                        float sqr = sin(qr) / qr;
                        float dsqr = cos(qr) - sqr;
                        float prefac = FF_kj * dsqr / r / r;
                        prefac += prefac;
                        S_calccs += FF_kj * sqr;
                        f_ptxcs += prefac * dx;
                        f_ptycs += prefac * dy;
                        f_ptzcs += prefac * dz;*/
                    //}
                }    
            }
        
            /*for (int stride = 512; stride > 128; stride >>= 1) {
                __syncthreads();
                for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                    q_a_r[iAccum] += q_a_r[iAccum + stride];
                    q_a_rx[iAccum] += q_a_rx[iAccum + stride];
                    q_a_ry[iAccum] += q_a_ry[iAccum + stride];
                    q_a_rz[iAccum] += q_a_rz[iAccum + stride];
                }
            }*/
            //__syncthreads();

            float S_calccs = 0.0;
            float f_ptxcs = 0.0;
            float f_ptycs = 0.0;
            float f_ptzcs = 0.0;

            for (int kk = 0; kk < 416; kk ++) {
                float r = (float) kk * 0.5 + 0.25;
                float qr = q_pt * r;
                float sqr = sin(qr) / qr;
                float dsqr = cos(qr) - sqr;
                float csqrr = dsqr / r / r;
                //atomicAdd(&S_calcc[ii*num_atom2+jj],(float)( 1e-6 * q_a_r[kk]  *   sqr[kk]));
                //atomicAdd(&f_ptxc[ii*num_atom2+jj], (float)( 1e-6 * q_a_rx[kk] * csqrr[kk]));
                //atomicAdd(&f_ptyc[ii*num_atom2+jj], (float)( 1e-6 * q_a_ry[kk] * csqrr[kk]));
                //atomicAdd(&f_ptzc[ii*num_atom2+jj], (float)( 1e-6 * q_a_rz[kk] * csqrr[kk]));
                S_calccs += a_r_q [(ii % 896)*num_q2*416+kk*num_q2+jj] *   sqr;
                f_ptxcs  += a_r_qx[(ii % 896)*num_q2*416+kk*num_q2+jj] * csqrr;
                f_ptycs  += a_r_qy[(ii % 896)*num_q2*416+kk*num_q2+jj] * csqrr;
                f_ptzcs  += a_r_qz[(ii % 896)*num_q2*416+kk*num_q2+jj] * csqrr;
            }

            S_calcc_T[ii*num_q2+jj]+= S_calccs;
            f_ptxc_T[ii*num_q2+jj] += f_ptxcs ;
            f_ptyc_T[ii*num_q2+jj] += f_ptycs ;
            f_ptzc_T[ii*num_q2+jj] += f_ptzcs ;

        }
        
    }
    //if (blockIdx.x == 0 && threadIdx.x == 0) printf("\nS_calcc[0] = %.3f \n",S_calcc[0]);
}


__global__ void __launch_bounds__(1024,2) scat_calc_bin2 (
    float *coord, 
    int *Ele,
    float *q_S_ref_dS, 
    float *S_calc, 
    int num_atom,   
    int num_q,     
    int num_ele,   
    float *Aq, 
    float alpha,   
    float k_chi,    
    float sigma2,
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    float *S_calcc, 
    int num_atom2,
    float *FF_full,
    int *q_a_r,
    int num_atom_3 // should be num_atom round up to 3072
    /*float *q_a_rx,
    float *q_a_ry,
    float *q_a_rz*/) { 

    // q_a_r is a 3D matrix of dimension 15 (1x SM), 64 (buffer), 192 (r_bin)


    extern __shared__ int buffer[];
    int *hist = buffer; // Histogram and the indices
    int size_buf = 12288; // For Kepler, use full shared memory
    float *dxyzFF = (float *)buffer; // dx, dy, dz, FF of that partition of atoms


    for (int jj = blockIdx.x; jj < num_atom; jj += gridDim.x) {
        // For every atom j
        int idx = jj % 15;
        float atom1x = coord[3*jj+0];
        float atom1y = coord[3*jj+1];
        float atom1z = coord[3*jj+2];
        int this_round = 0;
        for (int kk = this_round * 3 * blockDim.x + threadIdx.x; kk < num_atom_3; kk += 3 * blockDim.x) {
            // Flush and initialize the histogram
            for (int ll = threadIdx.x; ll < size_buf; ll += blockDim.x) hist[ll] = -2;
            __syncthreads();
            if (threadIdx.x < 192) hist[threadIdx.x] = 1;
            __syncthreads();
            
            int idy = kk;
            for (int ll = 0; ll < 3; ll ++) {
                if (idy < num_atom) {
                    float dx = coord[3*idy+0] - atom1x;
                    float dy = coord[3*idy+1] - atom1y;
                    float dz = coord[3*idy+2] - atom1z;
                    float r = sqrt(dx*dx+dy*dy+dz*dz);
                    int idz = r * 2.0;
                    if (idz < 192) {
                        int hist_idx = atomicAdd(&hist[idz],1);
                        if (hist_idx < 63) {
                            hist[hist_idx * 192 + idz] = idy % 3072;
                        }
                    }
                }

                idy += blockDim.x;
            }


            __syncthreads(); 
            // Now the histogram is done, copy it to global memory
            
            for (int ll = threadIdx.x; ll < size_buf; ll += blockDim.x) {
                q_a_r[idx*64*192 + ll] = hist[ll];
                hist[ll] = 0;
            }
            __syncthreads();

            // load with coord
            for (int ll = threadIdx.x; ll < size_buf / 4; ll += blockDim.x) {
                if (3072 * this_round + ll < num_atom) {
                    dxyzFF[ll + 3 * blockDim.x] = coord[3*(3072*this_round+ll)+0] - atom1x;
                    dxyzFF[ll + 6 * blockDim.x] = coord[3*(3072*this_round+ll)+1] - atom1y; 
                    dxyzFF[ll + 9 * blockDim.x] = coord[3*(3072*this_round+ll)+2] - atom1z;
                    // We'll load FF later
                }
            }
            __syncthreads();
            for (int ii = 0; ii < num_q; ii++) {
                float atom1FF = FF_full[ii * num_atom2 + jj];
                for (int ll = threadIdx.x; ll < size_buf / 4; ll += blockDim.x) {
                    if (3072 * this_round + ll < num_atom) 
                        dxyzFF[ll] = FF_full[ii * num_atom2 + 3072 * this_round + ll];
                }
                __syncthreads();
                // Now we start to calculate the scattering
                float q_pt = q_S_ref_dS[ii];
                float dx = 0.0;
                float dy = 0.0;
                float dz = 0.0;
                float FF = 0.0;
                if (threadIdx.x < 960) {
                    for (int ll = threadIdx.x; ll < size_buf - 192; ll += 960) {
                        if (q_a_r[idx * 64 * 192 + ll + 192] >= 0) {
                            int idy = q_a_r[idx * 64 * 192 + ll + 192];
                            dx += dxyzFF[idy] * atom1FF * dxyzFF[idy + 3 * 960];
                            dy += dxyzFF[idy] * atom1FF * dxyzFF[idy + 6 * 960];
                            dz += dxyzFF[idy] * atom1FF * dxyzFF[idy + 9 * 960];
                            FF += dxyzFF[idy] * atom1FF;
                        }
                    }
                    float r = (float)(idy % 192) * 0.5 + 0.25;
                    float qr = q_pt * r;
                    float sqr = sin(qr) / qr;
                    float dsqr = cos(qr) - sqr;
                    float csqrr = dsqr / r / r;
                    atomicAdd(&S_calcc[ii * num_atom2 + jj], FF * sqr);
                    atomicAdd(&f_ptxc [ii * num_atom2 + jj], dx * csqrr);
                    atomicAdd(&f_ptyc [ii * num_atom2 + jj], dy * csqrr);
                    atomicAdd(&f_ptzc [ii * num_atom2 + jj], dz * csqrr);

                }
                __syncthreads();

            }
           

        }
    this_round++;
    }
}

__global__ void sum_S_calc (
    float *S_calcc,
    float *f_ptxc,
    float *f_ptyc,
    float *f_ptzc,
    float *S_calc,
    float *Aq,
    float *q_S_ref_dS,
    int num_q,
    int num_atom,
    int num_atom2,
    float alpha,
    float k_chi,
    float *sigma2) {

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        S_calc[ii] = S_calcc[ii * num_atom2];
        __syncthreads();
        if (threadIdx.x == 0) {
            Aq[ii] = S_calc[ii] - q_S_ref_dS[ii+num_q];
            Aq[ii] *= -alpha;
            Aq[ii] += q_S_ref_dS[ii + 2*num_q];
            Aq[ii] *= k_chi / sigma2[ii];
            Aq[ii] += Aq[ii];
        }
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
            f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
        }
    }
}    

__global__ void force_calc (
    float *Force,
    int num_atom, 
    int num_q, 
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    int num_atom2, 
    int num_q2, 
    int *Ele,
    float force_ramp) {
    // Do column tree sum of f_ptxc for f_ptx for every atom, then assign threadIdx.x == 0 (3 * num_atoms) to Force. Force is num_atom * 3. 
    if (blockIdx.x >= num_atom) return;
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        for (int stride = num_q2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                f_ptxc[ii + iAccum * num_atom2] += f_ptxc[ii + iAccum * num_atom2 + stride * num_atom2];
                f_ptyc[ii + iAccum * num_atom2] += f_ptyc[ii + iAccum * num_atom2 + stride * num_atom2];
                f_ptzc[ii + iAccum * num_atom2] += f_ptzc[ii + iAccum * num_atom2 + stride * num_atom2];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (Ele[ii]) {
                Force[ii*3    ] = f_ptxc[ii] * force_ramp;
                Force[ii*3 + 1] = f_ptyc[ii] * force_ramp;
                Force[ii*3 + 2] = f_ptzc[ii] * force_ramp;
            }
        }
        __syncthreads();
    }
}


__global__ void force_calc_EMA (
    float *Force,
    double *Force_old, 
    int num_atom, 
    int num_q, 
    float *f_ptxc, 
    float *f_ptyc, 
    float *f_ptzc, 
    int num_atom2, 
    int num_q2, 
    int *Ele,
    double EMA_norm,
    float force_ramp) {
    // Do column tree sum of f_ptxc for f_ptx for every atom, then assign threadIdx.x == 0 (3 * num_atoms) to Force. Force is num_atom * 3. 
    if (blockIdx.x >= num_atom) return;
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        for (int stride = num_q2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                f_ptxc[ii + iAccum * num_atom2] += f_ptxc[ii + iAccum * num_atom2 + stride * num_atom2];
                f_ptyc[ii + iAccum * num_atom2] += f_ptyc[ii + iAccum * num_atom2 + stride * num_atom2];
                f_ptzc[ii + iAccum * num_atom2] += f_ptzc[ii + iAccum * num_atom2 + stride * num_atom2];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (Ele[ii]) {
                Force_old[ii*3    ] *= (EMA_norm - 1.0); 
                Force_old[ii*3    ] -= (double)f_ptxc[ii];
                Force_old[ii*3    ] /= EMA_norm;
                Force_old[ii*3 + 1] *= (EMA_norm - 1.0); 
                Force_old[ii*3 + 1] -= (double)f_ptyc[ii];
                Force_old[ii*3 + 1] /= EMA_norm;
                Force_old[ii*3 + 2] *= (EMA_norm - 1.0); 
                Force_old[ii*3 + 2] -= (double)f_ptzc[ii];
                Force_old[ii*3 + 2] /= EMA_norm;
                Force[ii*3    ] = (float)Force_old[ii*3    ] * force_ramp;
                Force[ii*3 + 1] = (float)Force_old[ii*3 + 1] * force_ramp;
                Force[ii*3 + 2] = (float)Force_old[ii*3 + 2] * force_ramp;
            }
        }
        __syncthreads();
    }
}
