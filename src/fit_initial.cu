
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <gsl/gsl_multimin.h>
//#include <gsl/gsl_vector.h>
#include "kernel.cu"
//#include "speedtest.hh"
#include "env_param.hh"
#include "mol_param.hh"
//#include "scat_param.hh"
#include "coord_ref.hh"
#include "expt_data.hh"
//#include "raster8.hh"

double my_f (const gsl_vector *v, void *params) {
    // The objective function chi square
    double c, a;
    double chi_sq = 0.0;
    c = gsl_vector_get(v, 0);
    a = gsl_vector_get(v, 1);
    double *p = (double *)params; 
    int num_q = (int)p[0];
    
    for (int ii = 0; ii < num_q; ii++) {
        chi_sq += (p[3*ii+1] - c * p[3*ii+2] + a) * 
                  (p[3*ii+1] - c * p[3*ii+2] + a) 
                  / p[3*ii+3] / p[3*ii+3];
    }

    return chi_sq;
}

void my_df (const gsl_vector *v, void *params, gsl_vector *df) {
    // The derivative of objective function chi square
    double c, a, dc, da;
    c = gsl_vector_get(v, 0);
    a = gsl_vector_get(v, 1);
    dc = 0.0;
    da = 0.0;
    double *p = (double *)params;
    int num_q = (int)p[0];
    for (int ii = 0; ii < num_q; ii++) {
        dc -= 2.0 * (p[3*ii+1] - c * p[3*ii+2] + a) * p[3*ii+2] / p[3*ii+3] / p[3*ii+3]; 
        //da += 2.0 * (p[3*ii+1] - c * p[3*ii+2] + a) / p[3*ii+3] / p[3*ii+3];
    }
    gsl_vector_set(df, 0, dc);
    gsl_vector_set(df, 1, da);
}

void my_fdf (const gsl_vector *x, void *params, double *f, gsl_vector *df) {
    *f= my_f(x, params);
    my_df(x, params, df);
}

double my_f_log (const gsl_vector *v, void *params) {
    // The objective function chi square
    double c, a;
    double chi_sq = 0.0;
    c = gsl_vector_get(v, 0);
    a = gsl_vector_get(v, 1);
    double *p = (double *)params; 
    int num_q = (int)p[0];
    
    for (int ii = 0; ii < num_q; ii++) {
        chi_sq += (p[3*ii+1] - p[3*ii+2] - c) * 
                  (p[3*ii+1] - p[3*ii+2] - c) 
                  / p[3*ii+3] / p[3*ii+3];
    }

    return chi_sq;
}

void my_df_log (const gsl_vector *v, void *params, gsl_vector *df) {
    // The derivative of objective function chi square
    double c, a, dc, da;
    c = gsl_vector_get(v, 0);
    a = gsl_vector_get(v, 1);
    dc = 0.0;
    da = 0.0;
    double *p = (double *)params;
    int num_q = (int)p[0];
    for (int ii = 0; ii < num_q; ii++) {
        dc -= 2.0 * (p[3*ii+1] - p[3*ii+2] - c) / p[3*ii+3] / p[3*ii+3]; 
        //da += 2.0 * (p[3*ii+1] - c * p[3*ii+2] + a) / p[3*ii+3] / p[3*ii+3];
    }
    gsl_vector_set(df, 0, dc);
    gsl_vector_set(df, 1, da);
}

void my_fdf_log (const gsl_vector *x, void *params, double *f, gsl_vector *df) {
    *f= my_f(x, params);
    my_df(x, params, df);
}

float fit_S_exp_to_S_calc(float *S_calc, float *S_exp, float *S_err, float *c, float *a, int num_q, int idx) {
    // This minimizes the function chi^2 = ( (S_calc - c * S_exp + a) / S_exp_err ) ^ 2
    // and returns the minimized chi^2.
    size_t iter = 0;
    int status;
    float chi_sq;

    //printf("In fitting.\n"); 
    
    double *p = (double *)malloc(sizeof(double) * (3 * num_q +1));

    p[0] = (double)num_q;
    for (int ii = 0; ii < num_q; ii++) {
        p[3*ii+1] = (double)S_calc[ii];
        p[3*ii+2] = (double)S_exp[ii];
        p[3*ii+3] = (double)S_err[ii];
    }
        
    gsl_vector *x; 
    gsl_multimin_function_fdf my_func;
    my_func.n = 2;
    my_func.f = &my_f;
    my_func.df = &my_df;
    my_func.fdf = &my_fdf;
    my_func.params = (void *)p;


    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

    x = gsl_vector_alloc(2);
    gsl_vector_set (x, 0, S_calc[0] / S_exp[0]);

    gsl_vector_set (x, 1, 0.0);

    T = gsl_multimin_fdfminimizer_conjugate_fr;
    s = gsl_multimin_fdfminimizer_alloc (T, 2);

    gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-6);

    do {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate (s);
        
        if (status) break;

        status = gsl_multimin_test_gradient (s->gradient, 1e-5); 
  
        /*if (status == GSL_SUCCESS) printf("Minimum found at:\n");
        
        printf ("%5d, %.5e, %.5e, %.5e\n", iter,
                gsl_vector_get (s->x, 0),
                gsl_vector_get (s->x, 1),
                s->f);*/
       } while (status == GSL_CONTINUE && iter < 1000);

    c[idx] = gsl_vector_get (s -> x, 0);
    a[idx] = gsl_vector_get (s -> x, 1);
    chi_sq = s->f;
    float S_calc_sum = 0.0;
    for (int ii = 0; ii < num_q; ii++) {
        S_calc_sum += S_calc[ii] * S_calc[ii];
    }

    gsl_multimin_fdfminimizer_free (s);
    gsl_vector_free (x);

    return (float)chi_sq / S_calc_sum;
}


float fit_S_exp_to_S_calc_log(float *S_calc, float *S_exp, float *S_err, float *c, float *a, int num_q, int idx) {
    // This minimizes the function chi^2 = ( (S_calc - c * S_exp + a) / S_exp_err ) ^ 2
    // and returns the minimized chi^2.
    size_t iter = 0;
    int status;
    float chi_sq;

    //printf("In fitting.\n"); 
    
    double *p = (double *)malloc(sizeof(double) * (3 * num_q +1));

    p[0] = (double)num_q;
    for (int ii = 0; ii < num_q; ii++) {
        p[3*ii+1] = log10((double)S_calc[ii]);
        p[3*ii+2] = log10((double)S_exp[ii]);
        double err_h = log10((double)S_exp[ii]+S_err[ii]);
        double err_l = log10((double)S_exp[ii]-S_err[ii]);
        p[3*ii+3] = (err_h - err_l) / 2.0;
    }
 
    gsl_vector *x; 
    gsl_multimin_function_fdf my_func;
    my_func.n = 2;
    my_func.f = &my_f_log;
    my_func.df = &my_df_log;
    my_func.fdf = &my_fdf_log;
    my_func.params = (void *)p;


    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

    x = gsl_vector_alloc(2);
    gsl_vector_set (x, 0, log10(S_calc[0] / S_exp[0]));
    gsl_vector_set (x, 1, 0.0);

    T = gsl_multimin_fdfminimizer_conjugate_fr;
    s = gsl_multimin_fdfminimizer_alloc (T, 2);

    gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-6);

    do {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate (s);
        
        if (status) break;

        status = gsl_multimin_test_gradient (s->gradient, 1e-5); 
  
        /*if (status == GSL_SUCCESS) printf("Minimum found at:\n");
        
        printf ("%5d, %.5e, %.5e, %.5e\n", iter,
                gsl_vector_get (s->x, 0),
                gsl_vector_get (s->x, 1),
                s->f);*/
    } while (status == GSL_CONTINUE && iter < 1000);

    c[idx] = gsl_vector_get (s -> x, 0);
    a[idx] = gsl_vector_get (s -> x, 1);
    chi_sq = s->f;
    float S_calc_sum = 0.0;
    for (int ii = 0; ii < num_q; ii++) {
        S_calc_sum += log10(S_calc[ii]) * log(S_calc[ii]);
    }

    gsl_multimin_fdfminimizer_free (s);
    gsl_vector_free (x);

    return (float)chi_sq / S_calc_sum;
}


int main (int argc, char* argv[]) {
// argv[1]: Beginning of c1 range
// argv[2]: Step of c1 scan
// argv[3]: Ending of c1 range
// argv[4]: Beginning of c2 range
// argv[5]: Step of c2 scan
// argv[6]: Ending of c2 range
    cudaFree(0);
    float *S_calc;

    // Declare cuda pointers //
    float *d_coord;          // Coordinates 3 x num_atom
    float *d_Force;          // Force 3 x num_atom
    int   *d_Ele;              // Element list.

    float *d_q_S_ref_dS;     /* q vector, reference scattering pattern, and 
                                measured difference pattern to fit.
                                Since they are of same size they're grouped */
                                
    float *d_sigma2;         // Sigma square (standard error of mean) for the target diff pattern.
    float *d_Aq;             // Prefactor for each q
    float *d_S_calc;         // Calculated scattering curve

    float *d_S_calcc,        // Some intermediate matrices
          *d_f_ptxc, 
          *d_f_ptyc, 
          *d_f_ptzc;
    
    float *d_V,              // Exposed surf area (in num of dots) 
          *d_V_s;            // Exposed surf area (in real A^2)

    float *d_WK;             // Waasmaier-Kirfel parameters 

    int   *d_close_flag,     // Flags for atoms close to an atom
          *d_close_num,      // Num of atoms close to an atom
          *d_close_idx;      // Their atomic index
 
    float *d_vdW;            // van der Waals radii

    float *d_FF_table,       // Form factors for each atom type at each q
          *d_FF_full;        /* Form factors for each atom at each q, 
                                considering the SASA an atom has. */
    int   *d_q_a_r;
  
    /*      *d_q_a_rx,
          *d_q_a_ry,
          *d_q_a_rz;*/


    // If using HyPred mode, then an array of c2 is needed. //
    float *d_c2_H;
    
    // set various memory chunk sizes
    int size_coord       = 3 * num_atom * sizeof(float);
    int size_atom        = num_atom * sizeof(int);
    int size_atom2       = num_atom2 * sizeof(int);
    int size_atom2f      = num_atom2 * sizeof(float);
    int size_atom2xatom2 = 1024 * num_atom2 * sizeof(int); // For d_close_flag
    int size_q           = num_q * sizeof(float); 
    int size_qxatom2     = num_q2 * num_atom2 * sizeof(float);
    int size_FF_table    = (num_ele + 1) * num_q * sizeof(float); // +1 for solvent
    int size_WK          = 11 * num_ele * sizeof(float);
    int size_vdW         = (num_ele + 1) * sizeof(float); // +1 for solvent
    int size_c2          = 10 * sizeof(float); // Only for HyPred
    int size_q_a_r       = 15 * 64 * 192 * sizeof(int);

    // Allocate local memories
    S_calc = (float *)malloc(size_q);

    /*for (int ii = 0; ii < num_q; ii++) {
        S_exp[ii] = q_S_ref_dS[ii+num_q];
    }*/

    // Allocate cuda memories
    cudaMalloc((void **)&d_Aq,         size_q);
    cudaMalloc((void **)&d_coord,      size_coord); // 40 KB
    cudaMalloc((void **)&d_Force,      size_coord); // 40 KB
    cudaMalloc((void **)&d_Ele,        size_atom);
    cudaMalloc((void **)&d_q_S_ref_dS, 3 * size_q);
    cudaMalloc((void **)&d_sigma2,     size_q);
    cudaMalloc((void **)&d_S_calc,     size_q); // Will be computed on GPU
    cudaMalloc((void **)&d_f_ptxc,     size_qxatom2);
    cudaMalloc((void **)&d_f_ptyc,     size_qxatom2);
    cudaMalloc((void **)&d_f_ptzc,     size_qxatom2);
    cudaMalloc((void **)&d_S_calcc,    size_qxatom2);
    cudaMalloc((void **)&d_V,          size_atom2f);
    cudaMalloc((void **)&d_V_s,        size_atom2f);
    cudaMalloc((void **)&d_close_flag, size_atom2xatom2);
    cudaMalloc((void **)&d_close_num,  size_atom2);
    cudaMalloc((void **)&d_close_idx,  size_atom2xatom2);
    cudaMalloc((void **)&d_vdW,        size_vdW);
    cudaMalloc((void **)&d_FF_table,   size_FF_table);
    cudaMalloc((void **)&d_FF_full,    size_qxatom2);
    cudaMalloc((void **)&d_WK,         size_WK);
    cudaMalloc((void **)&d_c2_H,       size_c2); // Only for HyPred
    cudaMalloc((void **)&d_q_a_r,      size_q_a_r);
    /*cudaMalloc((void **)&d_q_a_rx,      size_q_a_r);
    cudaMalloc((void **)&d_q_a_ry,      size_q_a_r);
    cudaMalloc((void **)&d_q_a_rz,      size_q_a_r);*/

    // Initialize some matrices
    cudaMemset(d_close_flag, 0,   size_qxatom2);
    cudaMemset(d_Force,      0.0, size_coord);
    cudaMemset(d_Aq,         0.0, size_q);
    cudaMemset(d_S_calc,     0.0, size_q);
    cudaMemset(d_f_ptxc,     0.0, size_qxatom2);
    cudaMemset(d_f_ptyc,     0.0, size_qxatom2);   
    cudaMemset(d_f_ptzc,     0.0, size_qxatom2);
    cudaMemset(d_S_calcc,    0.0, size_qxatom2);
    cudaMemset(d_close_num,  0,   size_atom2);
    cudaMemset(d_close_idx,  0,   size_atom2xatom2);
    cudaMemset(d_FF_full,    0.0, size_qxatom2);
    cudaMemset(d_q_a_r,      0.0, size_q_a_r);
    /*cudaMemset(d_q_a_rx,     0.0, size_q_a_r);
    cudaMemset(d_q_a_ry,     0.0, size_q_a_r);
    cudaMemset(d_q_a_rz,     0.0, size_q_a_r);*/
    // Copy necessary data
    cudaMemcpy(d_coord,      coord_init, size_coord, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vdW,        vdW,        size_vdW,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,        Ele,        size_atom,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma2,     dS_err,     size_q,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_S_ref_dS, q,          size_q,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,         WK,         size_WK,    cudaMemcpyHostToDevice);
    // Only for HyPred
    cudaMemcpy(d_c2_H,       c2_H,       size_c2,    cudaMemcpyHostToDevice);



    float sigma2 = 1.0;
    float alpha = 1.0;

    /*float c1_init   = 0.96;
    float c1_step   = 0.02;
    float c1_end    = 1.06;*/
    float c1_init   = atof(argv[1]);
    float c1_step   = atof(argv[2]);
    float c1_end    = atof(argv[3]);
    /*float c1_init   = 1.00;
    float c1_step   = 0.002;
    float c1_end    = 1.00;*/
    int c1_step_num = (int)((c1_end - c1_init) / c1_step + 1.0);
    float c2_init   = atof(argv[4]);
    float c2_step   = atof(argv[5]);
    float c2_end    = atof(argv[6]);
    /*float c2_init   = 2.0;
    float c2_step   = 0.1;
    float c2_end    = 2.0;*/
    int c2_step_num = (int)((c2_end - c2_init) / c2_step + 1.0);
    int use_log     = 0;
    int outputFile  = 1;
    float *chi_sq = (float *)malloc(c1_step_num * c2_step_num * sizeof(float));
    float *c = (float *)malloc(c1_step_num * c2_step_num * sizeof(float));
    float *a = (float *)malloc(c1_step_num * c2_step_num * sizeof(float));

    dist_calc<<<1024, 1024>>>(
        d_coord, 
        d_close_num, 
        d_close_flag,
        d_close_idx, 
        num_atom,
        num_atom2); 

    surf_calc<<<1024,512>>>(
        d_coord, 
        d_Ele, 
        d_close_num, 
        d_close_idx, 
        d_vdW, 
        num_atom, 
        num_atom2, 
        num_raster, 
        sol_s, 
        d_V);


    /*sum_V<<<1,1024>>>(
        d_V, 
        d_V_s, 
        num_atom, 
        num_atom2, 
        d_Ele, 
        d_vdW);*/
    
    /*printf("%d ", num_q);
    printf("%d ", c1_step_num); 
    for (int ii = 0; ii < num_q; ii++) {
        printf("%.6f ", q[ii]);
    }
    printf("\n");*/
    

    float c1_fit = c1_init;
    int idx = 0;
    int min_idx = 0;
    float min_chi2 = FLT_MAX; 
    float min_c;
    float min_a;
    float min_c1;
    float min_c2;
    float *min_S_calc = (float *)malloc(num_q * sizeof(float));
    for (int ii = 0; ii < c1_step_num; ii++) {
        float c2_F_fit = c2_init;
        FF_calc<<<320, 32>>>(
            d_q_S_ref_dS, 
            d_WK, 
            d_vdW, 
            num_q, 
            num_ele, 
            c1_fit, 
            r_m, 
            d_FF_table,
            rho);

        for (int jj = 0; jj < c2_step_num; jj++) {
/*            create_FF_full_HyPred<<<320, 1024>>>(
                d_FF_table, 
                d_V,
                c2_F,
                d_c2_H, 
                d_Ele, 
                d_FF_full, 
                num_q, 
                num_ele, 
                num_atom, 
                num_atom2);*/

            create_FF_full_FoXS<<<320, 1024>>>(
                d_FF_table, 
                d_V,
                c2_F_fit,
                d_Ele, 
                d_FF_full, 
                num_q, 
                num_ele, 
                num_atom, 
                num_atom2);
        
            scat_calc<<<320, 1024>>>(
                d_coord, 
                d_Ele,
                d_q_S_ref_dS, 
                d_S_calc, 
                num_atom,  
                num_q,     
                num_ele,  
                d_Aq, 
                alpha,    
                k_chi,     
                d_sigma2,    
                d_f_ptxc, 
                d_f_ptyc, 
                d_f_ptzc, 
                d_S_calcc, 
                num_atom2, 
                d_FF_full);


            cudaMemcpy(S_calc, d_S_calc, size_q,     cudaMemcpyDeviceToHost);
            
            // Need to do some sort of fitting. 
            if (use_log) {
                chi_sq[idx] = fit_S_exp_to_S_calc_log(S_calc, S_exp, S_err, c, a, num_q, idx);
            } else {
                chi_sq[idx] = fit_S_exp_to_S_calc(S_calc, S_exp, S_err, c, a, num_q, idx);
            }
            //printf("c1 = %.3f, c2_F = %.3f ", c1, c2_F);
            //printf("c = %.3e, a = %.3e ", c[idx], a[idx]);
            //printf("chi square = %.5f\n", chi_sq[idx]);
            if (chi_sq[idx] < min_chi2) {
                min_chi2 = chi_sq[idx];
                min_idx = idx;
                min_c = c[idx];
                min_a = a[idx];
                min_c1 = c1_fit;
                min_c2 = c2_F_fit;
                for (int kk = 0; kk < num_q; kk++) {
                    min_S_calc[kk] = S_calc[kk];
                }
            }
            /*printf("%.3f %.3f ", c1, c2_F);
            for (int kk = 0; kk < num_q; kk++) {
                printf("%.5f ",S_calc[kk]);
            }
            printf("\n");*/

            if (outputFile) {
                // Output
                printf("#Fitting result: \n");
                printf("#c1 = %.3f, c2 = %.3f, c = %.3e, a = %.3e, chi2 = %.3e \n", 
                            c1_fit,  c2_F_fit,   c[idx],   a[idx], chi_sq[idx]);
            
                FILE *fp = fopen("scat_param_log.txt","a");
                fprintf(fp, "\n//SCAT_PARAM_LOG_%.3f_%.3f", c1_fit, c2_F_fit);
                fprintf(fp, "\n#include \"scat_param.hh\"\n\n");
                fprintf(fp, "int num_q = %d;\n", num_q);
                fprintf(fp, "int num_q2 = %d;\n", (num_q+31)/32*32);
                fprintf(fp, "float c1 = %.3f;\n", c1_fit);
                fprintf(fp, "float c2 = %.3f;\n", c2_F_fit);
                fprintf(fp, "float c = %.3f;\n", c[idx]);
                fprintf(fp, "float q_S_ref_dS[%d] = {", 3 * num_q);
                for (int ii = 0; ii < num_q; ii++) {
                    fprintf(fp, "%.5f", q[ii]);
                    fprintf(fp,", ");
                }
                fprintf(fp, "\n");
                for (int ii = 0; ii < num_q; ii++) {
                    fprintf(fp, "%.5f", S_calc[ii]);
                    fprintf(fp,", ");
                }
                fprintf(fp, "\n");
                double sum_potential = 0.0;
                for (int ii = 0; ii < num_q; ii++) {
                    if (use_log) {
                        fprintf(fp, "%.5f", dS_exp[ii] * pow(10.0,c[idx]) - a[idx]);
                        sum_potential += (S_exp[ii] * pow(10.0,c[idx]) - a[idx])*(S_exp[ii] * pow(10.0,c[idx]) - a[idx]) / dS_err[ii];
                    } else {
                        fprintf(fp, "%.5f", dS_exp[ii] * c[idx] - a[idx]);
                        sum_potential += (dS_exp[ii] * c[idx] - a[idx]) * (dS_exp[ii] * c[idx] - a[idx]) / dS_err[ii];
                    }
                    if (ii < num_q -1) fprintf(fp,", ");
                }
                fprintf(fp, "};\n//");
                 
                for (int ii = 0; ii < num_q; ii++) {
                    if (use_log) {
                        fprintf(fp, "%.5f", S_exp[ii] * pow(10.0,c[idx]) + a[idx]);
                    } else {
                        fprintf(fp, "%.5f", S_exp[ii] * c[idx] + a[idx]);
                    }
                    if (ii < num_q -1) fprintf(fp,", ");
                }
                fprintf(fp, "\n");
                fprintf(fp, "float dS_err[%d] = {", num_q);
                for (int ii = 0; ii < num_q; ii++) {
                    fprintf(fp, "%.5f", dS_err[ii] * c[idx]);
                    if (ii < num_q - 1) fprintf(fp, ", ");
                }
                fprintf(fp, "};\n\n");
            
                fprintf(fp, "//SCAT_KCHI_%.3f_%.3f\n", c1_fit, c2_F_fit);
                fprintf(fp, "float k_chi = %.3e;\n", 28.0 / sum_potential);
                fprintf(fp, "sum of potential is %.3e\n", sum_potential);
                fclose(fp);
                fp = fopen("scat_param_log.hh","a");
                fprintf(fp, "\n//SCAT_PARAM_LOG_%.3f_%.3f\n", c1_fit, c2_F_fit);
                fprintf(fp, "extern int num_q;\n");
                fprintf(fp, "extern int num_q2;\n");
                fprintf(fp, "extern float c1;\n");
                fprintf(fp, "extern float c2;\n");
                fprintf(fp, "extern float c;\n");
                fprintf(fp, "extern float q_S_ref_dS[%d];\n", 3 * num_q);
                fprintf(fp, "extern float dS_err[%d];\n", num_q);
                
                fclose(fp);
            
                fp = fopen("scat_param_log.dat","a");
            
                fprintf(fp,"data = [");
                for (int ii = 0; ii < num_q; ii++) {
                    fprintf(fp, "%.5f", q[ii]);
                    fprintf(fp,", ");
                }
                fprintf(fp, "\n");
                for (int ii = 0; ii < num_q; ii++) {
                    fprintf(fp, "%.5f", S_calc[ii]);
                    fprintf(fp,", ");
                }
                fprintf(fp, "\n");
                for (int ii = 0; ii < num_q; ii++) {
                    if (use_log) {
                        fprintf(fp, "%.5f", S_exp[ii] * pow(10.0,c[idx]) + a[idx] - S_calc[ii]);
                    } else {
                        fprintf(fp, "%.5f", S_exp[ii] * c[idx] + a[idx] - S_calc[ii]);
                    }
                    fprintf(fp,", ");
                }
                fprintf(fp, "\n");
                 
                for (int ii = 0; ii < num_q; ii++) {
                    if (use_log) {
                        fprintf(fp, "%.5f", S_exp[ii] * pow(10.0,c[idx]) + a[idx]);
                    } else {
                        fprintf(fp, "%.5f", S_exp[ii] * c[idx] + a[idx]);
                    }
                    fprintf(fp,", ");
                }
                fprintf(fp, "\n");
                for (int ii = 0; ii < num_q; ii++) {
                    if (use_log) {
                        fprintf(fp, "%.5f", S_err[ii] * pow(10.0,c[idx]) + a[idx]);
                    } else {
                        fprintf(fp, "%.5f", S_err[ii] * c[idx] + a[idx]);
                    }
                    if (ii < num_q-1) fprintf(fp,", ");
                }
                fprintf(fp, "];\n");
                fclose(fp);
                            
            }

 
            idx++;
            printf("Now processing idx %d\n",idx); 
            // Initialize some matrices
            cudaMemset(d_close_flag, 0,   size_qxatom2);
            cudaMemset(d_Force,      0.0, size_coord);
            cudaMemset(d_Aq,         0.0, size_q);
            cudaMemset(d_S_calc,     0.0, size_q);
            cudaMemset(d_f_ptxc,     0.0, size_qxatom2);
            cudaMemset(d_f_ptyc,     0.0, size_qxatom2);   
            cudaMemset(d_f_ptzc,     0.0, size_qxatom2);
            cudaMemset(d_S_calcc,    0.0, size_qxatom2);
            cudaMemset(d_close_num,  0,   size_atom2);
            cudaMemset(d_close_idx,  0,   size_atom2xatom2);
            cudaMemset(d_FF_full,    0.0, size_qxatom2);
            cudaMemset(d_q_a_r,      0.0, size_q_a_r);
            /*cudaMemset(d_q_a_rx,     0.0, size_q_a_r);
            cudaMemset(d_q_a_ry,     0.0, size_q_a_r);
            cudaMemset(d_q_a_rz,     0.0, size_q_a_r);*/
            
            c2_F_fit += c2_step;
        }

        c1_fit += c1_step;
    }
    

    // Output
    printf("#Final fitting result: \n");
    printf("#c1 = %.3f, c2 = %.3f, c = %.3e, a = %.3e, chi2 = %.3e \n", 
                min_c1,    min_c2,    min_c,    min_a,    min_chi2);

    FILE *fp = fopen("scat_param.cu","w");
    fprintf(fp, "\n#include \"scat_param.hh\"\n\n");
    fprintf(fp, "int num_q = %d;\n", num_q);
    fprintf(fp, "int num_q2 = %d;\n", (num_q+31)/32*32);
    fprintf(fp, "float c1 = %.3f;\n", min_c1);
    fprintf(fp, "float c2 = %.3f;\n", min_c2);
    fprintf(fp, "float c = %.3f;\n", min_c);
    /*fprintf(fp, "float c1 = %.3f;\n", 1.000);
    fprintf(fp, "float c2 = %.3f;\n", 2.000);
    fprintf(fp, "float c = %.3f;\n", 1.000);*/
    fprintf(fp, "float q_S_ref_dS[%d] = {", 3 * num_q);
    for (int ii = 0; ii < num_q; ii++) {
        fprintf(fp, "%.5f", q[ii]);
        fprintf(fp,", ");
    }
    fprintf(fp, "\n");
    for (int ii = 0; ii < num_q; ii++) {
        fprintf(fp, "%.5f", min_S_calc[ii]);
        fprintf(fp,", ");
    }
    fprintf(fp, "\n");
    for (int ii = 0; ii < num_q; ii++) {
        if (use_log) {
            fprintf(fp, "%.5f", dS_exp[ii] * pow(10.0,min_c) + min_a - min_S_calc[ii]);
        } else {
            fprintf(fp, "%.5f", dS_exp[ii] * min_c + min_a - min_S_calc[ii]);
        }
        if (ii < num_q -1) fprintf(fp,", ");
    }
    fprintf(fp, "};\n//");
     
    for (int ii = 0; ii < num_q; ii++) {
        if (use_log) {
            fprintf(fp, "%.5f", S_exp[ii] * pow(10.0,min_c) + min_a);
        } else {
            fprintf(fp, "%.5f", S_exp[ii] * min_c + min_a);
        }
        if (ii < num_q -1) fprintf(fp,", ");
    }
    fprintf(fp, "\n");
    fprintf(fp, "float dS_err[%d] = {", num_q);
    for (int ii = 0; ii < num_q; ii++) {
        fprintf(fp, "%.5f", dS_err[ii] * min_c);
        if (ii < num_q - 1) fprintf(fp, ", ");
    }
    fprintf(fp, "};\n");

    fclose(fp);
    fp = fopen("scat_param.hh","w");
    fprintf(fp, "extern int num_q;\n");
    fprintf(fp, "extern int num_q2;\n");
    fprintf(fp, "extern float c1;\n");
    fprintf(fp, "extern float c2;\n");
    fprintf(fp, "extern float c;\n");
    fprintf(fp, "extern float q_S_ref_dS[%d];\n", 3 * num_q);
    fprintf(fp, "extern float dS_err[%d];\n", num_q);
    
    fclose(fp);

    fp = fopen("scat_param.dat","w");

    fprintf(fp,"data = [");
    for (int ii = 0; ii < num_q; ii++) {
        fprintf(fp, "%.5f", q[ii]);
        fprintf(fp,", ");
    }
    fprintf(fp, "\n");
    for (int ii = 0; ii < num_q; ii++) {
        fprintf(fp, "%.5f", min_S_calc[ii]);
        fprintf(fp,", ");
    }
    fprintf(fp, "\n");
    for (int ii = 0; ii < num_q; ii++) {
        if (use_log) {
            fprintf(fp, "%.5f", S_exp[ii] * pow(10.0,min_c) + min_a - min_S_calc[ii]);
        } else {
            fprintf(fp, "%.5f", S_exp[ii] * min_c + min_a - min_S_calc[ii]);
        }
        fprintf(fp,", ");
    }
    fprintf(fp, "\n");
     
    for (int ii = 0; ii < num_q; ii++) {
        if (use_log) {
            fprintf(fp, "%.5f", S_exp[ii] * pow(10.0,min_c) + min_a);
        } else {
            fprintf(fp, "%.5f", S_exp[ii] * min_c + min_a);
        }
        fprintf(fp,", ");
    }
    fprintf(fp, "\n");
    for (int ii = 0; ii < num_q; ii++) {
        if (use_log) {
            fprintf(fp, "%.5f", S_err[ii] * pow(10.0,min_c) + min_a);
        } else {
            fprintf(fp, "%.5f", S_err[ii] * min_c + min_a);
        }
        if (ii < num_q-1) fprintf(fp,", ");
    }
    fprintf(fp, "];\n");
    fclose(fp);


 
    cudaFree(d_coord); 
    cudaFree(d_Force); 
    cudaFree(d_Ele); 
    cudaFree(d_q_S_ref_dS); 
    cudaFree(d_Aq);
    cudaFree(d_S_calc); 
    cudaFree(d_f_ptxc); cudaFree(d_f_ptyc); cudaFree(d_f_ptzc);
    cudaFree(d_S_calcc); 
    cudaFree(d_WK);
    cudaFree(d_V); cudaFree(d_V_s); 
    cudaFree(d_close_flag); cudaFree(d_close_num); cudaFree(d_close_idx);
    cudaFree(d_vdW);
    cudaFree(d_FF_table); cudaFree(d_FF_full);
    cudaFree(d_c2_H);
    cudaFree(d_q_a_r);// cudaFree(d_q_a_rx); cudaFree(d_q_a_ry); cudaFree(d_q_a_rz);
    free(S_calc); free(min_S_calc);

    return 0;
}
