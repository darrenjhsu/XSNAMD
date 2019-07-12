
#include "coord_ref.hh"

float coord_ref[312] = {-0.132, 2.45, -0.112, 0.22, 3.108, -0.747, -0.767, 3.067, 0.352, -0.701, 1.208, -0.598, -0.992, 0.67, 0.193, -1.903, 1.542, -1.431, -2.108, 2.634, -1.418, -1.743, 1.237, -2.489, -2.803, 1.018, -1.044, 0.301, 0.419, -1.423, 0.479, -0.783, -1.242, 0.969, 1.064, -2.333, 0.833, 2.036, -2.485, 1.956, 0.379, -3.172, 1.468, -0.304, -3.715, 2.588, 1.376, -4.097, 3.693, 1.256, -4.12, 2.208, 1.245, -5.134, 2.362, 2.413, -3.766, 3.021, -0.294, -2.323, 3.374, -1.452, -2.536, 3.552, 0.402, -1.362, 3.271, 1.339, -1.184, 4.589, -0.17, -0.499, 5.382, -0.373, -1.074, 4.982, 0.845, 0.533, 5.946, 0.573, 1.014, 5.101, 1.849, 0.07, 4.208, 0.912, 1.328, 4.097, -1.439, 0.176, 4.783, -2.459, 0.205, 2.921, -1.408, 0.73, 2.36, -0.588, 0.71, 2.374, -2.587, 1.406, 2.967, -2.79, 2.185, 0.998, -2.267, 1.911, 0.95, -1.232, 2.313, 0.25, -2.351, 1.092, 0.709, -2.966, 2.726, 2.321, -3.779, 0.466, 2.725, -4.888, 0.812, 1.827, -3.587, -0.721, 1.5, -2.694, -1.008, 1.736, -4.685, -1.687, 1.112, -5.368, -1.309, 1.164, -4.16, -2.971, 1.507, -3.121, -3.167, 1.482, -4.793, -3.829, 0.053, -4.153, -2.93, 3.099, -5.306, -1.938, 3.26, -6.525, -1.934, 4.095, -4.5, -2.161, 3.973, -3.514, -2.165, 5.442, -5.017, -2.417, 5.406, -5.547, -3.264, 6.382, -3.863, -2.602, 7.428, -4.158, -2.369, 6.354, -3.5, -3.652, 6.105, -3.022, -1.929, 5.914, -5.897, -1.272, 6.435, -6.992, -1.476, 5.751, -5.448, -0.063, 5.331, -4.565, 0.111, 6.183, -6.232, 1.098, 7.177, -6.331, 1.042, 5.836, -5.483, 2.351, 6.748, -5.08, 2.841, 5.163, -4.628, 2.123, 5.324, -6.153, 3.076, 5.524, -7.601, 1.108, 6.173, -8.625, 1.312, 4.242, -7.651, 0.896, 3.711, -6.828, 0.729, 3.527, -8.931, 0.893, 3.604, -9.321, 1.81, 2.08, -8.682, 0.587, 1.806, -7.623, 0.784, 1.862, -8.899, -0.481, 1.432, -9.328, 1.219, 4.121, -9.884, -0.13, 4.369, -11.055, 0.152, 4.355, -9.419, -1.321, 4.158, -8.473, -1.556, 4.922, -10.273, -2.368, 4.261, -11.001, -2.553, 5.113, -9.459, -3.614, 6.188, -9.24, -3.788, 4.727, -10.005, -4.502, 4.574, -8.49, -3.533, 6.247, -10.87, -1.926, 6.494, -12.065, -2.078, 7.115, -10.07, -1.383, 6.922, -9.103, -1.256, 8.419, -10.564, -0.933, 8.912, -10.893, -1.74, 9.188, -9.431, -0.32, 10.153, -9.785, 0.102, 9.412, -8.653, -1.082, 8.622, -8.953, 0.509, 8.258, -11.691, 0.074, 8.906, -12.731, -0.017, 7.577, -11.424, 0.756, 7.471, -12.165, 1.418, 7.24, -10.508, 0.537};
float coord_init[312] = {-0.132, 2.45, -0.112, 0.22, 3.108, -0.747, -0.767, 3.067, 0.352, -0.701, 1.208, -0.598, -0.992, 0.67, 0.193, -1.903, 1.542, -1.431, -2.108, 2.634, -1.418, -1.743, 1.237, -2.489, -2.803, 1.018, -1.044, 0.301, 0.419, -1.423, 0.479, -0.783, -1.242, 0.969, 1.064, -2.333, 0.833, 2.036, -2.485, 1.956, 0.379, -3.172, 1.468, -0.304, -3.715, 2.588, 1.376, -4.097, 3.693, 1.256, -4.12, 2.208, 1.245, -5.134, 2.362, 2.413, -3.766, 3.021, -0.294, -2.323, 3.374, -1.452, -2.536, 3.552, 0.402, -1.362, 3.271, 1.339, -1.184, 4.589, -0.17, -0.499, 5.382, -0.373, -1.074, 4.982, 0.845, 0.533, 5.946, 0.573, 1.014, 5.101, 1.849, 0.07, 4.208, 0.912, 1.328, 4.097, -1.439, 0.176, 4.783, -2.459, 0.205, 2.921, -1.408, 0.73, 2.36, -0.588, 0.71, 2.374, -2.587, 1.406, 2.967, -2.79, 2.185, 0.998, -2.267, 1.911, 0.95, -1.232, 2.313, 0.25, -2.351, 1.092, 0.709, -2.966, 2.726, 2.321, -3.779, 0.466, 2.725, -4.888, 0.812, 1.827, -3.587, -0.721, 1.5, -2.694, -1.008, 1.736, -4.685, -1.687, 1.112, -5.368, -1.309, 1.164, -4.16, -2.971, 1.507, -3.121, -3.167, 1.482, -4.793, -3.829, 0.053, -4.153, -2.93, 3.099, -5.306, -1.938, 3.26, -6.525, -1.934, 4.095, -4.5, -2.161, 3.973, -3.514, -2.165, 5.442, -5.017, -2.417, 5.406, -5.547, -3.264, 6.382, -3.863, -2.602, 7.428, -4.158, -2.369, 6.354, -3.5, -3.652, 6.105, -3.022, -1.929, 5.914, -5.897, -1.272, 6.435, -6.992, -1.476, 5.751, -5.448, -0.063, 5.331, -4.565, 0.111, 6.183, -6.232, 1.098, 7.177, -6.331, 1.042, 5.836, -5.483, 2.351, 6.748, -5.08, 2.841, 5.163, -4.628, 2.123, 5.324, -6.153, 3.076, 5.524, -7.601, 1.108, 6.173, -8.625, 1.312, 4.242, -7.651, 0.896, 3.711, -6.828, 0.729, 3.527, -8.931, 0.893, 3.604, -9.321, 1.81, 2.08, -8.682, 0.587, 1.806, -7.623, 0.784, 1.862, -8.899, -0.481, 1.432, -9.328, 1.219, 4.121, -9.884, -0.13, 4.369, -11.055, 0.152, 4.355, -9.419, -1.321, 4.158, -8.473, -1.556, 4.922, -10.273, -2.368, 4.261, -11.001, -2.553, 5.113, -9.459, -3.614, 6.188, -9.24, -3.788, 4.727, -10.005, -4.502, 4.574, -8.49, -3.533, 6.247, -10.87, -1.926, 6.494, -12.065, -2.078, 7.115, -10.07, -1.383, 6.922, -9.103, -1.256, 8.419, -10.564, -0.933, 8.912, -10.893, -1.74, 9.188, -9.431, -0.32, 10.153, -9.785, 0.102, 9.412, -8.653, -1.082, 8.622, -8.953, 0.509, 8.258, -11.691, 0.074, 8.906, -12.731, -0.017, 7.577, -11.424, 0.756, 7.471, -12.165, 1.418, 7.24, -10.508, 0.537};
