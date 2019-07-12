%module XSMD
%{
    #include "XSMD.hh"
    #include "mol_param.hh"
    #include "env_param.hh"
    #include "scat_param.hh"
    #include "WaasKirf.hh"
    #define PI 3.14159265359
%}

// SWIG helper functions for arrays
%inline %{
/* Create an array */
float *float_array(int size) {
    return (float *)malloc(size*sizeof(float));
}
/* Get a value from an array */
float float_get(float *a, int index) {
    return a[index];
}
/* Set a value in the array */
float float_set(float *a, int index, float value) {
    return (a[index] = value);
}

void float_destroy(float *a) { 
    free(a);
}
/* Create an array */
double *double_array(int size) {
    return (double *)malloc(size*sizeof(double));
}
/* Get a value from an array */
double double_get(double *a, int index) {
    return a[index];
}
/* Set a value in the array */
double double_set(double *a, int index, double value) {
    return (a[index] = value);
}

void double_destroy(double *a) { 
    free(a);
}
%}
%include "XSMD.hh"
%include "mol_param.hh"
%include "env_param.hh"
%include "scat_param.hh"
%include "WaasKirf.hh"
%define PI 3.14159265359
%enddef
