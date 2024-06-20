#ifndef __SNN_LIF_HPP
#define __SNN_LIF_HPP

#include "ap_int.h"

const float dt[30]={0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03};

#include "weights1.hpp"
#include "weights2.hpp"

struct axis_in {
float data;
ap_uint<1> last;
};

struct axis_out {
float data;
ap_uint<1> last;
};

float clip(float n, float lower, float upper);
void update_voltage(float* train, float* nodes, int n, float t, float* refractory_time);
void m_multiply1(float train_new[64], float train[784]);
void m_multiply2(float train_new2[10], float train[64]);
void snn_lif(axis_in *inp, axis_out *spike_sum);

#endif
