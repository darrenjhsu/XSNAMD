
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "kernel.cu"
#include "traj_scatter.hh"
#include "mol_param.hh" // So we have Ele to use
#include "scat_param.hh" // So we have q to use 
#include "env_param.hh" // So we have c2 to use
#include "WaasKirf.hh"

int main(int argc, char* argv[]) {

// Calculate scattering for every frame in trajectory
// argv[1]: name of the file
// argv[2]: number of frames

// Parameters
    //int num_q = 98;
    //int num_atom = 3649;
    int frames_to_average = atoi(argv[2]); // Will be the last n frames of the traj
    int frames_total = atoi(argv[2]); // Look at the xyz file

    float *coord, *S_calc, *S_calc_tot;
    float *d_coord, *d_S_calc;
    int *d_Ele;
    float *d_q_S_ref_dS, *d_Force, *d_FF, *d_sigma2;
    float *d_Aq;
    float *d_S_calcc, *d_f_ptxc, *d_f_ptyc, *d_f_ptzc;
    float *d_raster, *d_V, *d_V_s;
    float *d_WK;
    int *d_close_flag, *d_close_num, *d_close_idx;
    float *d_vdW;
    int *close_num, *close_idx;
    float *V;
    float *d_FF_table, *d_FF_full;
    float *d_c2;

    int size_coord = 3 * num_atom * sizeof(float);
    int size_atom = num_atom * sizeof(int);
    int size_atom2 = num_atom2 * sizeof(int);
    int size_atom2f = num_atom2 * sizeof(float);
    int size_atomxatom = num_atom * num_atom * sizeof(float);
    int size_atom2xatom2 = 1024 * num_atom2 * sizeof(int);
    int size_q = num_q * sizeof(float); 
    int size_FF = num_ele * num_q * sizeof(float);
    int size_qxatom2 = num_q2 * num_atom2 * sizeof(float); // check if overflow
    int size_FF_table = (num_ele+1) * num_q * sizeof(float);
    int size_surf = num_atom * num_raster * 3 * sizeof(float);
    int size_WK = 11 * num_ele * sizeof(float);
    int size_vdW = (num_ele+1) * sizeof(float);
    int size_c2 = 10 * sizeof(float);

    // Allocate cuda memories
    cudaMalloc((void **)&d_Aq,     size_q);
    cudaMalloc((void **)&d_coord,  size_coord); // 40 KB
    cudaMalloc((void **)&d_Force,  size_coord); // 40 KB
    cudaMalloc((void **)&d_Ele,    size_atom);
    cudaMalloc((void **)&d_q_S_ref_dS, 3 * size_q);
    cudaMalloc((void **)&d_sigma2,     size_q);
    cudaMalloc((void **)&d_S_calc, size_q); // Will be computed on GPU
    cudaMalloc((void **)&d_f_ptxc, size_qxatom2);
    cudaMalloc((void **)&d_f_ptyc, size_qxatom2);
    cudaMalloc((void **)&d_f_ptzc, size_qxatom2);
    cudaMalloc((void **)&d_S_calcc, size_qxatom2);
    cudaMalloc((void **)&d_V, size_atom2f);
    cudaMalloc((void **)&d_V_s, size_atom2f);
    cudaMalloc((void **)&d_close_flag, size_atom2xatom2);
    cudaMalloc((void **)&d_close_num, size_atom2);
    cudaMalloc((void **)&d_close_idx, size_atom2xatom2);
    cudaMalloc((void **)&d_vdW, size_vdW);
    cudaMalloc((void **)&d_FF_table, size_FF_table);
    cudaMalloc((void **)&d_FF_full, size_qxatom2);
    cudaMalloc((void **)&d_WK, size_WK);
    cudaMalloc((void **)&d_c2, size_c2);
 
    // Allocate local memory
    coord = (float *)malloc(size_coord);
    S_calc = (float *)malloc(size_q);
    S_calc_tot = (float *)malloc(size_q);
    char* buf[100], buf1[100], buf2[100], buf3[100];
    float f1, f2, f3;
    V = (float *)malloc(size_atom2f);
    // Initialize cuda matrices
    cudaMemset(d_Aq, 0.0, size_q);
    cudaMemset(d_S_calc, 0.0, size_q);
    cudaMemset(d_f_ptxc,0.0, size_qxatom2);
    cudaMemset(d_f_ptyc,0.0, size_qxatom2);   
    cudaMemset(d_f_ptzc,0.0, size_qxatom2);
    cudaMemset(d_S_calcc,0.0, size_qxatom2);
    cudaMemset(d_close_flag, 0, size_qxatom2);
    cudaMemset(d_close_num, 0, size_atom2);
    cudaMemset(d_close_idx, 0, size_atom2xatom2);
    // Copy necessary data
//    cudaMemcpy(d_coord, coord, size_coord,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_vdW, vdW, size_vdW, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,    Ele,    size_atom,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_S_ref_dS,  q_S_ref_dS, 3 * size_q,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma2,     dS_err,     size_q,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,     WK,     size_WK,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2,   c2_H,     size_c2,     cudaMemcpyHostToDevice);


    // Initialize local matrices
    for (int ii = 0; ii < 3 * num_atom; ii++) coord[ii] = 0.0;
    for (int ii = 0; ii < num_q; ii++) { 
        S_calc[ii] = 0.0;
        S_calc_tot[ii] = 0.0;
    }

    float sigma2 = 1.0;
    float alpha = 1.0;

    //FILE *fp = fopen("aligned_protein.xyz","r");
    FILE *fp = fopen(argv[1],"r");
    if (fp == NULL) {
        printf("Opening file failed.\n");
        return 1;
    } else {
        printf("Opened file.\n");
    }
    // Read file by num_atom
    for (int ii = 0; ii < frames_total; ii++) {
        fscanf(fp,"%*s",buf);
        fscanf(fp,"%*s %d",buf);
        printf("Read the first two lines, ii = %d\n", ii);
        for (int jj = 0; jj < num_atom; jj++) {
            fscanf(fp,"%s %f %f %f",buf, &f1, &f2, &f3);
            //printf("Readed line %d\n", jj);
            coord[3*jj] = f1;
            coord[3*jj+1] = f2;
            coord[3*jj+2] = f3;
            //printf("Coord[jj] = %.3f, Coord[jj+1] = %.3f, Coord[jj+2] = %.3f\n",coord[3*jj], coord[3*jj+1], coord[3*jj+2]);
        }
        printf("First coordinate is: %.3f\n",coord[0]);
        printf("Final coordinate is: %.3f\n",coord[3*num_atom-1]);
        if (ii >= frames_total - frames_to_average) {
            printf("Calculating frame %d...\n", ii);
            cudaMemcpy(d_coord, coord, size_coord, cudaMemcpyHostToDevice);
            cudaMemset(d_Aq, 0.0, size_q);
            cudaMemset(d_S_calc, 0.0, size_q);
            cudaMemset(d_f_ptxc,0.0, size_qxatom2);
            cudaMemset(d_f_ptyc,0.0, size_qxatom2);   
            cudaMemset(d_f_ptzc,0.0, size_qxatom2);
            cudaMemset(d_S_calcc,0.0, size_qxatom2);
            cudaMemset(d_close_flag, 0, size_qxatom2);
            cudaMemset(d_close_num, 0, size_atom2);
            cudaMemset(d_close_idx, 0, size_atom2xatom2);
            dist_calc<<<1024, 1024>>>(d_coord, //d_dx, d_dy, d_dz, 
                                      d_close_num, d_close_flag, d_close_idx, num_atom, num_atom2); 
            surf_calc<<<1024,512>>>(d_coord, d_Ele, d_close_num, d_close_idx, d_vdW, 
                                    num_atom, num_atom2, num_raster, sol_s, d_V);
            // Output V
            cudaMemcpy(V, d_V, size_atom2f, cudaMemcpyDeviceToHost);
            
            printf("V for frame %5d, ",ii);
            for (int jj = 0; jj < num_atom; jj++) {
                printf("%6.3f", V[jj]);
                if (jj < num_atom - 1) {
                    printf(", ");
                }
            }
            printf("\n");

            sum_V<<<1,1024>>>(d_V, d_V_s, num_atom, num_atom2, d_Ele, d_vdW);
            FF_calc<<<320, 32>>>(d_q_S_ref_dS, d_WK, d_vdW, num_q, num_ele, c1, r_m, d_FF_table, rho);
            //create_FF_full_HyPred<<<320, 1024>>>(d_FF_table, d_V, c2, d_c2, d_Ele, d_FF_full, 
            //                              num_q, num_ele, num_atom, num_atom2);
            create_FF_full_FoXS<<<320, 1024>>>(d_FF_table, d_V, c2, d_Ele, d_FF_full, 
                                          num_q, num_ele, num_atom, num_atom2);
            scat_calc<<<320, 1024>>>(d_coord,    
                                     d_Ele,        
                                     d_q_S_ref_dS, 
                                     d_S_calc, num_atom,  num_q,     num_ele,  d_Aq, 
                                     alpha,    k_chi,     d_sigma2,    d_f_ptxc, d_f_ptyc, 
                                     d_f_ptzc, d_S_calcc, num_atom2, 
                                     d_FF_full);
            cudaMemcpy(S_calc ,d_S_calc, size_q,     cudaMemcpyDeviceToHost);
            printf("S_calc for frame %5d, ",ii);
            for (int jj = 0; jj < num_q; jj++) {
                printf("%6.3f, ", S_calc[jj]);
                S_calc_tot[jj] += S_calc[jj];
            }
            printf("\n");
        }
    }
    fclose(fp);
    for (int ii = 0; ii < num_q; ii++) {
        S_calc_tot[ii] /= float(frames_to_average);
        printf("q = %.3f, S(q) = %.5f \n", q_S_ref_dS[ii], S_calc_tot[ii]);
    }


    // Free cuda and local memories
    cudaFree(d_coord); cudaFree(d_Force); 
    cudaFree(d_Ele); 
    cudaFree(d_q_S_ref_dS); 
    cudaFree(d_S_calc); cudaFree(d_Aq);
    cudaFree(d_f_ptxc); cudaFree(d_f_ptyc); cudaFree(d_f_ptzc);
    cudaFree(d_S_calcc); cudaFree(d_WK);
    cudaFree(d_V); 
    cudaFree(d_close_flag); cudaFree(d_close_num); cudaFree(d_close_idx);
    cudaFree(d_vdW);

    return 0;
}
