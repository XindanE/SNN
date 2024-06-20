#include "snn_lif.hpp"
#include <algorithm>
#include <cmath>

// Network architecture and parameters
int layers[3] = {784, 64, 10};
int T = 30;
float pot_rest = 0;
float pot_thr = 1;
float tau_ref = 0.002;

// Nodes and refractory time initialization
float nodes0[784] = {0};
float nodes1[64] = {0};
float nodes2[10] = {0};

float refractory_time0[784] = {0.002};
float refractory_time1[64] = {0.002};
float refractory_time2[10] = {0.002};

// 量化参数
const float scale_input = 255.0;
const float inv_scale_input = 1.0 / scale_input;
const float scale_weight = 127.0;
const float inv_scale_weight = 1.0 / scale_weight;
const float scale_output = 1023.0;
const float inv_scale_output = 1.0 / scale_output;

// Function to clip values within a range
float clip(float n, float lower, float upper) {
    return std::max(lower, std::min(n, upper));
}

// 量化函数
ap_uint<16> quantize(float value, float scale) {
    return static_cast<ap_uint<16>>(clip(value * scale, 0.0f, scale));
}

// 反量化函数
float dequantize(ap_uint<16> value, float inv_scale) {
    return value * inv_scale;
}

// First layer matrix multiplication
void m_multiply1(ap_uint<16> train_new[64], ap_uint<16> train[784]) {
    float res[64] = {0};
    for (int i = 0; i < 64; i++) {
        res[i] = 0;
        for (int j = 0; j < 784; j++) {
#pragma HLS PIPELINE
            res[i] += dequantize(train[j], inv_scale_input) * weights1[i][j];
        }
    }
    for (int i = 0; i < 64; i++) {
        train_new[i] = quantize(res[i], scale_output);
    }
}

// Second layer matrix multiplication
void m_multiply2(ap_uint<16> train_new2[10], ap_uint<16> train[64]) {
    float res[10] = {0};
    for (int i = 0; i < 10; i++) {
        res[i] = 0;
        for (int j = 0; j < 64; j++) {
#pragma HLS PIPELINE
            res[i] += dequantize(train[j], inv_scale_output) * weights2[i][j];
        }
    }
    for (int i = 0; i < 10; i++) {
        train_new2[i] = quantize(res[i], scale_output);
    }
}

// Update the voltage of neurons
void update_voltage(ap_uint<16> train[784], float* nodes, int n, float t, float* refractory_time) {
    float tau_rc = 0.02;
    float delta, nu, spike_mask, t_spike;

    for (int i = 0; i < n; i++) {
        refractory_time[i] -= t;
        nu = t - refractory_time[i];
        delta = clip(nu, 0, t);
        nodes[i] -= (dequantize(train[i], inv_scale_input) - nodes[i]) * std::expm1f(-delta / tau_rc);

        if (nodes[i] > 1) train[i] = quantize(1 / t, scale_input);
        float eps = std::pow(10, -9);
        t_spike = t + tau_rc * std::log1pf(-1 * (nodes[i] - 1) / (dequantize(train[i], inv_scale_input) - 1) + eps);
        if (nodes[i] < 0) nodes[i] = 0;

        refractory_time[i] = tau_ref + t_spike;
    }
}

// Main SNN LIF function
void snn_lif(ap_uint<8> inp[784], ap_uint<16> spike_sum[10]){
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL
#pragma HLS INTERFACE s_axilite port=inp bundle=CONTROL
#pragma HLS INTERFACE s_axilite port=spike_sum bundle=CONTROL

    int spikes[30][10] = {0};
    ap_uint<16> train0[784] = {0};
#pragma HLS ARRAY_PARTITION variable=train0 cyclic factor=4 dim=1
    ap_uint<16> train1[64] = {0};
#pragma HLS ARRAY_PARTITION variable=train1 cyclic factor=4 dim=1
    ap_uint<16> train2[10] = {0};

    // 重置节点电压和重置时间
    std::fill_n(nodes0, 784, pot_rest);
    std::fill_n(nodes1, 64, pot_rest);
    std::fill_n(nodes2, 10, pot_rest);
    std::fill_n(refractory_time0, 784, tau_ref);
    std::fill_n(refractory_time1, 64, tau_ref);
    std::fill_n(refractory_time2, 10, tau_ref);

    // Process each time step
    for (int t = 0; t < 30; t++) {
        for (int ind5 = 0; ind5 < 784; ind5++) {
#pragma HLS PIPELINE II=1
            train0[ind5] = quantize(inp[ind5], scale_input);
        }

        update_voltage(train0, nodes0, 784, dt[t], refractory_time0);

        for (int sp = 0; sp < 784; sp++) {
            if (nodes0[sp] > pot_thr) nodes0[sp] = pot_rest;
        }

        m_multiply1(train1, train0);
        update_voltage(train1, nodes1, 64, dt[t], refractory_time1);

        for (int j = 0; j < 64; j++) {
            if (nodes1[j] > pot_thr) nodes1[j] = pot_rest;
        }

        m_multiply2(train2, train1);
        update_voltage(train2, nodes2, 10, dt[t], refractory_time2);

        for (int k = 0; k < 10; k++) {
            if (nodes2[k] > pot_thr) {
                nodes2[k] = pot_rest;
                spikes[t][k] = 1;
            } else {
                spikes[t][k] = 0;
            }
        }
    }

    // Sum spikes over all time steps
    for (int sn = 0; sn < 10; sn++) {
        int tot = 0;
        for (int sm = 0; sm < 30; sm++) {
            tot += spikes[sm][sn];
        }
        spike_sum[sn] = tot;
    }
}
