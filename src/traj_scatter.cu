
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "kernel.cu"
#include "env_param.cu"
#include "traj_scatter.hh"
#include "WaasKirf.cu"
#include "vdW.cu"

int main(int argc, char* argv[]) {

// Calculate scattering for every frame in trajectory
// argv[1]: name of the xyz file - we'll parse the number of atoms and atomic number from that
// argv[2]: name of the q file (format: num_q q q q q ...)
// argv[3]: c1
// argv[4]: c2

// Parameters

    float *coord, *S_calc, *S_calc_tot;
    int *Ele;
    float *d_coord, *d_S_calc;
    float *d_q;
    int *d_Ele;
    float *d_FF;
    float *d_S_calcc;
    float *d_raster, *d_V, *d_V_s;
    float *d_WK;
    int *d_close_flag, *d_close_num, *d_close_idx;
    float *d_vdW;
    int *close_num, *close_idx;
    float *V;
    float *d_FF_table, *d_FF_full;

    char* buf[100], buf1[100], buf2[100], buf3[100];
    float f1, f2, f3;
    int num_q, num_atom, buf_d;

    // Process the q file
    FILE *fp = fopen(argv[2],"r");
    if (fp == NULL) {
        printf("Opening q file failed.\n");
        return 1;
    } else {
        printf("Opened q file.\n");
    }
    fscanf(fp,"%d",&num_q);
    float* q;
    q = (float *)malloc(num_q * sizeof(float));
    //printf("Number of q points = %d\n", num_q);
    for (int jj = 0; jj < num_q; jj++) {
        fscanf(fp,"%f",&f1);
        //printf("Readed line %d\n", jj);
        q[jj] = f1;
        //printf("%f, ", q[jj]);
    }
    //printf("\n");
    fclose(fp);
    
    // Parse xyz file for Ele and num_atom
    fp = fopen(argv[1],"r");
    if (fp == NULL) {
        printf("Opening file failed.\n");
        return 1;
    } else {
        printf("Opened file.\n");
    }

    fscanf(fp,"%d",&num_atom);
    fscanf(fp,"%*s %d",buf, &buf_d);
    

    int num_atom2 = (num_atom + 2047) / 2048 * 2048;
    //printf("num_atom = %d, num_atom2 = %d\n", num_atom, num_atom2);
    int size_coord = 3 * num_atom * sizeof(float);
    int size_atom = num_atom * sizeof(int);

    coord = (float*)malloc(size_coord);
    Ele = (int*)malloc(size_atom);  

    for (int jj = 0; jj < num_atom; jj++) {
        fscanf(fp,"%d %f %f %f", &buf_d, &f1, &f2, &f3);
        //printf("Readed line %d\n", jj);
        Ele[jj] = buf_d-1;
        coord[3*jj] = f1;
        coord[3*jj+1] = f2;
        coord[3*jj+2] = f3;
        //printf("Coord[jj] = %.3f, Coord[jj+1] = %.3f, Coord[jj+2] = %.3f\n",coord[3*jj], coord[3*jj+1], coord[3*jj+2]);
    }
    fclose(fp);
    /*
    for (int jj = 0; jj < num_atom; jj++) {
        printf("%d, %f, %f, %f\n", Ele[jj], coord[3*jj], coord[3*jj+1], coord[3*jj+2]);
    }
    */
    float c1 = atof(argv[3]);
    float c2 = atof(argv[4]);

    //printf("c1 and c2 are %.3f and %.3f\n", c1, c2);

    int size_atom2 = num_atom2 * sizeof(int);
    int size_atom2f = num_atom2 * sizeof(float);
    int size_atomxatom = num_atom * num_atom * sizeof(float);
    int size_atom2xatom2 = 1024 * num_atom2 * sizeof(int);
    int size_q = num_q * sizeof(float); 
    int size_FF = num_ele * num_q * sizeof(float);
    int size_qxatom2 = num_q * num_atom2 * sizeof(float); // check if overflow
    int size_FF_table = (num_ele+1) * num_q * sizeof(float);
    int size_surf = num_atom * num_raster * 3 * sizeof(float);
    int size_WK = 11 * num_ele * sizeof(float);
    int size_vdW = (num_ele+1) * sizeof(float);

    // Allocate cuda memories
    cudaMalloc((void **)&d_coord,  size_coord); // 40 KB
    cudaMalloc((void **)&d_Ele,    size_atom);
    cudaMalloc((void **)&d_q,    size_q);
    cudaMalloc((void **)&d_S_calc, size_q); // Will be computed on GPU
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
 
    // Allocate local memory
    coord = (float *)malloc(size_coord);
    S_calc = (float *)malloc(size_q);
    S_calc_tot = (float *)malloc(size_q);
    V = (float *)malloc(size_atom2f);
    // Initialize cuda matrices
    cudaMemset(d_S_calc, 0.0, size_q);
    cudaMemset(d_S_calcc,0.0, size_qxatom2);
    cudaMemset(d_close_flag, 0, size_qxatom2);
    cudaMemset(d_close_num, 0, size_atom2);
    cudaMemset(d_close_idx, 0, size_atom2xatom2);
    // Copy necessary data
    cudaMemcpy(d_q,        q,    size_q,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_vdW,    vdW,    size_vdW,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,    Ele,    size_atom,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,     WK,     size_WK,     cudaMemcpyHostToDevice);


    // Initialize local matrices
    for (int ii = 0; ii < 3 * num_atom; ii++) coord[ii] = 0.0;
    for (int ii = 0; ii < num_q; ii++) { 
        S_calc[ii] = 0.0;
        S_calc_tot[ii] = 0.0;
    }

    // Read file by num_atom
    fp = fopen(argv[1],"r");
    for (int ii = 0; ii < 1; ii++) {
        fscanf(fp,"%*s",buf);
        fscanf(fp,"%*s %d",buf, &buf_d);
        //printf("Read the first two lines, ii = %d\n", ii);
        for (int jj = 0; jj < num_atom; jj++) {
            fscanf(fp,"%s %f %f %f",buf, &f1, &f2, &f3);
            //printf("Readed line %d\n", jj);
            coord[3*jj] = f1;
            coord[3*jj+1] = f2;
            coord[3*jj+2] = f3;
            //printf("Coord[jj] = %.3f, Coord[jj+1] = %.3f, Coord[jj+2] = %.3f\n",coord[3*jj], coord[3*jj+1], coord[3*jj+2]);
        }
        //printf("First coordinate is: %.3f\n",coord[0]);
        //printf("Final coordinate is: %.3f\n",coord[3*num_atom-1]);
        //printf("Calculating frame %d...\n", ii);
        cudaMemcpy(d_coord, coord, size_coord, cudaMemcpyHostToDevice);
        cudaMemset(d_S_calc, 0.0, size_q);
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
       
        /* 
        printf("V for frame %5d, ",ii);
        for (int jj = 0; jj < num_atom; jj++) {
            printf("%6.3f", V[jj]);
            if (jj < num_atom - 1) {
                printf(", ");
            }
        }
        printf("\n");
        */

        sum_V<<<1,1024>>>(d_V, d_V_s, num_atom, num_atom2, d_Ele, sol_s, d_vdW);
        FF_calc<<<num_q, 32>>>(d_q, d_WK, d_vdW, num_q, num_ele, c1, r_m, d_FF_table, rho);
       
        /* 
        float *FF_table = (float*)malloc(size_FF_table);
        cudaMemcpy(FF_table, d_FF_table, size_FF_table, cudaMemcpyDeviceToHost);
        printf("FF_table, \n");
        for (int jj = 0; jj < num_q; jj++) {
            for (int kk = 0; kk < num_ele+1; kk++) { 
                printf("%6.3f", FF_table[jj * (num_ele+1) + kk]);
                if (kk < num_ele - 1) {
                    printf(", ");
                }
            }
            printf("\n");
        }
        printf("\n");
        */

        create_FF_full_FoXS<<<num_q, 1024>>>(d_FF_table, d_V, c2, d_Ele, d_FF_full, 
                                      num_q, num_ele, num_atom, num_atom2);

        /* 
        float *FF_full = (float*)malloc(size_qxatom2);
        cudaMemcpy(FF_full, d_FF_full, size_qxatom2, cudaMemcpyDeviceToHost);
        printf("Full FF for frame %5d, ",ii);
        for (int jj = 0; jj < num_q; jj++) {
            for (int kk = 0; kk < num_atom; kk++) { 
                printf("%6.3f", FF_full[jj * num_atom2 + kk]);
                if (kk < num_atom - 1) {
                    printf(", ");
                }
            }
            printf("\n");
        }
        printf("\n");
        */

        pure_scat_calc<<<num_q, 1024>>>(d_coord, d_Ele, d_q,  
                                 d_S_calc, num_atom,  num_q, num_ele,
                                 d_S_calcc, num_atom2, 
                                 d_FF_full);
        cudaMemcpy(S_calc ,d_S_calc, size_q,     cudaMemcpyDeviceToHost);
        printf("S_calc for frame %5d, ",ii);
        for (int jj = 0; jj < num_q; jj++) {
            printf("%6.3f, ", S_calc[jj]);
            S_calc_tot[jj] += S_calc[jj];
        }
        printf("\n");
    }
    fclose(fp);
    for (int ii = 0; ii < num_q; ii++) {
        S_calc_tot[ii] /= float(100);
        //printf("q = %.3f, S(q) = %.5f \n", q[ii], S_calc_tot[ii]);
    }


    // Free cuda and local memories
    cudaFree(d_coord); 
    cudaFree(d_Ele); 
    cudaFree(d_q); 
    cudaFree(d_S_calc);
    cudaFree(d_S_calcc); 
    cudaFree(d_WK);
    cudaFree(d_V); 
    cudaFree(d_close_flag); cudaFree(d_close_num); cudaFree(d_close_idx);
    cudaFree(d_vdW);

    return 0;
}
