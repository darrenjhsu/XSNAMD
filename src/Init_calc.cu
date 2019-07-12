
#include <stdio.h>
#include <math.h>
#include "Init_calc.hh"
#include "coord_ref.hh"
#include "WaasKirf.hh"
#include "param.hh"
#define PI 3.14159265359

double FFcalc (double *WK, double q, int Ele) {
    double q_WK = q / 4.0 / PI;
    int jj = Ele;
    double FF = WK[jj*11] * exp(-WK[jj*11+6] * q_WK * q_WK) + \
                WK[jj*11+1] * exp(-WK[jj*11+7] * q_WK * q_WK) + \
                WK[jj*11+2] * exp(-WK[jj*11+8] * q_WK * q_WK) + \
                WK[jj*11+3] * exp(-WK[jj*11+9] * q_WK * q_WK) + \
                WK[jj*11+4] * exp(-WK[jj*11+10] * q_WK * q_WK) + \
                WK[jj*11+5];
    return FF;
}

int main () {
    double S_ref[num_q];
    double S_init[num_q];
//    for (int ii = 0; ii < num_atom; ii++) printf("%d ", Ele[ii]);
    for (int ii = 0; ii < num_q; ii++) {
        S_ref[ii] = 0.0;
        S_init[ii] = 0.0;
    }
    printf("double S_ref[%d] = {",num_q);
    for (int ii = 0; ii < num_q; ii++) {
//        printf("%d: %.5f = q. ", ii, q[ii]);
        for (int jj = 0; jj < num_atom; jj++) {
            int atom1t = Ele[jj];
//            printf("%d ",atom1t);
            double atom1FF = FFcalc(WK, q[ii], atom1t);
            double atom1x = coord_ref[3*jj+0];
            double atom1y = coord_ref[3*jj+1];
            double atom1z = coord_ref[3*jj+2];

            for (int kk = 0; kk < num_atom; kk++) {
                int atom2t = Ele[kk];
                double atom2FF = FFcalc(WK, q[ii], atom2t);
                if (q[ii] == 0.0) {
                    S_ref[ii] += atom1FF * atom2FF;
                } else if (kk == jj) {
                    S_ref[ii] += atom1FF * atom2FF;
                } else {
                    double dx = coord_ref[3*kk+0] - atom1x;
                    double dy = coord_ref[3*kk+1] - atom1y;
                    double dz = coord_ref[3*kk+2] - atom1z;
//                    if (kk == 0 && jj == 1) printf("\n%.5f, %.5f, %.5f, %.5f, %.5f, %.5f", coord_ref[3*jj+0],coord_ref[3*jj+1],coord_ref[3*jj+2],coord_ref[3*kk+0], coord_ref[3*kk+1], coord_ref[3*kk+2]);
                    double r = sqrt(dx*dx+dy*dy+dz*dz);
//                    if (kk == 0 && jj == 1) printf("\nkk = %d, jj = %d, r2 = %.5f, r = %.5f, FFsin/qr = %.5f \n", kk,jj,dx*dx+dy*dy+dz*dz,r, atom1FF * atom2FF * sin(q[ii] * r) / q[ii] / r);
                    S_ref[ii] += atom1FF * atom2FF * sin(q[ii] * r) / q[ii] / r;
                }
            }
        }
        printf("%.5f",S_ref[ii]);
        if (ii < num_q - 1) printf(", ");
    }
    printf("}; \n");
    printf("double dS[%d] = {",num_q);
    for (int ii = 0; ii < num_q; ii++) {
//        printf("%d: %.5f = q. ", ii, q[ii]);
        for (int jj = 0; jj < num_atom; jj++) {
            int atom1t = Ele[jj];
//            printf("%d ",atom1t);
            double atom1FF = FFcalc(WK, q[ii], atom1t);
            double atom1x = coord_init[3*jj+0];
            double atom1y = coord_init[3*jj+1];
            double atom1z = coord_init[3*jj+2];

            for (int kk = 0; kk < num_atom; kk++) {
                int atom2t = Ele[kk];
                double atom2FF = FFcalc(WK, q[ii], atom2t);
                if (q[ii] == 0.0) {
                    S_init[ii] += atom1FF * atom2FF;
                } else if (kk == jj) {
                    S_init[ii] += atom1FF * atom2FF;
                } else {
                    double dx = coord_init[3*kk+0] - atom1x;
                    double dy = coord_init[3*kk+1] - atom1y;
                    double dz = coord_init[3*kk+2] - atom1z;
//                    if (kk == 0 && jj == 1) printf("\n%.5f, %.5f, %.5f, %.5f, %.5f, %.5f", coord_init[3*jj+0],coord_init[3*jj+1],coord_init[3*jj+2],coord_init[3*kk+0], coord_init[3*kk+1], coord_init[3*kk+2]);
                    double r = sqrt(dx*dx+dy*dy+dz*dz);
//                    if (kk == 0 && jj == 1) printf("\nkk = %d, jj = %d, r2 = %.5f, r = %.5f, FFsin/qr = %.5f \n", kk,jj,dx*dx+dy*dy+dz*dz,r, atom1FF * atom2FF * sin(q[ii] * r) / q[ii] / r);
                    S_init[ii] += atom1FF * atom2FF * sin(q[ii] * r) / q[ii] / r;
                }
            }
        }
        printf("%.5f",S_ref[ii] - S_init[ii]);
        if (ii < num_q - 1) printf(", ");
    }
    printf("}; \n");


    return 0;
}

