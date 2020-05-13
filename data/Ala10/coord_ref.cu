
#include "coord_ref.hh"
//These are Ala10 random coil coordinates
float coord_ref[312] = {-0.132, 2.45, -0.112, 0.22, 3.108, -0.747, -0.767, 3.067, 0.352, -0.701, 1.208, -0.598, -1.037, 0.692, 0.19, -1.858, 1.545, -1.491, -2.05, 2.639, -1.501, -1.653, 1.224, -2.536, -2.782, 1.037, -1.14, 0.323, 0.385, -1.361, 1.392, 0.057, -0.853, 0.027, 0.038, -2.578, -0.836, 0.3, -2.997, 0.958, -0.756, -3.385, 1.105, -1.623, -2.909, 0.335, -1.033, -4.721, -0.467, -0.298, -4.949, 1.096, -0.969, -5.529, -0.115, -2.049, -4.739, 2.28, -0.029, -3.565, 2.967, -0.186, -4.572, 2.659, 0.772, -2.614, 2.109, 0.903, -1.796, 3.918, 1.515, -2.71, 3.862, 2.1, -3.519, 4.071, 2.378, -1.493, 5.054, 2.208, -1.002, 4.007, 3.455, -1.765, 3.274, 2.153, -0.752, 5.101, 0.569, -2.832, 6.21, 0.869, -2.394, 4.898, -0.575, -3.415, 4.004, -0.823, -3.773, 5.984, -1.546, -3.57, 6.304, -1.789, -2.654, 5.448, -2.777, -4.24, 4.34, -2.823, -4.165, 5.718, -2.784, -5.319, 5.863, -3.692, -3.765, 7.126, -0.963, -4.384, 7.842, -1.673, -5.087, 7.318, 0.321, -4.315, 6.744, 0.902, -3.749, 8.395, 0.965, -5.072, 8.226, 0.792, -6.043, 8.35, 2.444, -4.826, 9.335, 2.822, -4.48, 8.082, 2.989, -5.758, 7.596, 2.687, -4.046, 9.751, 0.41, -4.672, 10.767, 1.099, -4.722, 9.799, -0.825, -4.267, 8.981, -1.389, -4.225, 11.066, -1.437, -3.859, 11.42, -0.909, -3.086, 10.807, -2.845, -3.41, 9.773, -2.957, -3.02, 10.932, -3.555, -4.257, 11.513, -3.132, -2.601, 12.069, -1.427, -4.999, 12.926, -2.302, -5.108, 11.987, -0.459, -5.863, 11.297, 0.252, -5.782, 12.913, -0.378, -6.996, 12.796, -1.208, -7.542, 12.556, 0.813, -7.836, 13.37, 1.569, -7.824, 12.385, 0.51, -8.892, 11.632, 1.297, -7.452, 14.351, -0.268, -6.518, 15.201, 0.333, -7.172, 14.652, -0.831, -5.385, 13.971, -1.319, -4.851, 16.015, -0.768, -4.85, 16.241, 0.197, -4.715, 16.052, -1.468, -3.524, 15.053, -1.458, -3.037, 16.361, -2.529, -3.649, 16.772, -0.969, -2.84, 17.007, -1.408, -5.807, 18.029, -1.956, -5.4, 16.733, -1.359, -7.077, 15.909, -0.917, -7.413, 17.634, -1.955, -8.066, 17.699, -2.933, -7.864, 17.045, -1.778, -9.434, 16.927, -0.701, -9.683, 17.701, -2.237, -10.206, 16.043, -2.256, -9.492, 19.012, -1.317, -8.004, 19.724, -1.228, -9.002, 19.411, -0.862, -6.854, 18.841, -0.931, -6.042, 20.724, -0.228, -6.707, 20.742, 0.57, -7.309, 20.893, 0.227, -5.288, 21.779, -0.25, -4.818, 21.033, 1.329, -5.244, 20.01, -0.035, -4.666, 21.839, -1.19, -7.084, 22.949, -1.122, -6.56, 21.501, -1.823, -7.78, 22.235, -2.456, -8.028, 20.548, -1.603, -7.986};
float coord_init[312] = {-0.132, 2.45, -0.112, 0.22, 3.108, -0.747, -0.767, 3.067, 0.352, -0.701, 1.208, -0.598, -1.037, 0.692, 0.19, -1.858, 1.545, -1.491, -2.05, 2.639, -1.501, -1.653, 1.224, -2.536, -2.782, 1.037, -1.14, 0.323, 0.385, -1.361, 1.392, 0.057, -0.853, 0.027, 0.038, -2.578, -0.836, 0.3, -2.997, 0.958, -0.756, -3.385, 1.105, -1.623, -2.909, 0.335, -1.033, -4.721, -0.467, -0.298, -4.949, 1.096, -0.969, -5.529, -0.115, -2.049, -4.739, 2.28, -0.029, -3.565, 2.967, -0.186, -4.572, 2.659, 0.772, -2.614, 2.109, 0.903, -1.796, 3.918, 1.515, -2.71, 3.862, 2.1, -3.519, 4.071, 2.378, -1.493, 5.054, 2.208, -1.002, 4.007, 3.455, -1.765, 3.274, 2.153, -0.752, 5.101, 0.569, -2.832, 6.21, 0.869, -2.394, 4.898, -0.575, -3.415, 4.004, -0.823, -3.773, 5.984, -1.546, -3.57, 6.304, -1.789, -2.654, 5.448, -2.777, -4.24, 4.34, -2.823, -4.165, 5.718, -2.784, -5.319, 5.863, -3.692, -3.765, 7.126, -0.963, -4.384, 7.842, -1.673, -5.087, 7.318, 0.321, -4.315, 6.744, 0.902, -3.749, 8.395, 0.965, -5.072, 8.226, 0.792, -6.043, 8.35, 2.444, -4.826, 9.335, 2.822, -4.48, 8.082, 2.989, -5.758, 7.596, 2.687, -4.046, 9.751, 0.41, -4.672, 10.767, 1.099, -4.722, 9.799, -0.825, -4.267, 8.981, -1.389, -4.225, 11.066, -1.437, -3.859, 11.42, -0.909, -3.086, 10.807, -2.845, -3.41, 9.773, -2.957, -3.02, 10.932, -3.555, -4.257, 11.513, -3.132, -2.601, 12.069, -1.427, -4.999, 12.926, -2.302, -5.108, 11.987, -0.459, -5.863, 11.297, 0.252, -5.782, 12.913, -0.378, -6.996, 12.796, -1.208, -7.542, 12.556, 0.813, -7.836, 13.37, 1.569, -7.824, 12.385, 0.51, -8.892, 11.632, 1.297, -7.452, 14.351, -0.268, -6.518, 15.201, 0.333, -7.172, 14.652, -0.831, -5.385, 13.971, -1.319, -4.851, 16.015, -0.768, -4.85, 16.241, 0.197, -4.715, 16.052, -1.468, -3.524, 15.053, -1.458, -3.037, 16.361, -2.529, -3.649, 16.772, -0.969, -2.84, 17.007, -1.408, -5.807, 18.029, -1.956, -5.4, 16.733, -1.359, -7.077, 15.909, -0.917, -7.413, 17.634, -1.955, -8.066, 17.699, -2.933, -7.864, 17.045, -1.778, -9.434, 16.927, -0.701, -9.683, 17.701, -2.237, -10.206, 16.043, -2.256, -9.492, 19.012, -1.317, -8.004, 19.724, -1.228, -9.002, 19.411, -0.862, -6.854, 18.841, -0.931, -6.042, 20.724, -0.228, -6.707, 20.742, 0.57, -7.309, 20.893, 0.227, -5.288, 21.779, -0.25, -4.818, 21.033, 1.329, -5.244, 20.01, -0.035, -4.666, 21.839, -1.19, -7.084, 22.949, -1.122, -6.56, 21.501, -1.823, -7.78, 22.235, -2.456, -8.028, 20.548, -1.603, -7.986};