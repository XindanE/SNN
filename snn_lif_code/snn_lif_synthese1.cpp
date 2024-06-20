#include "snn_lif.hpp"
#include <algorithm> // for std::max and std::min
#include <cmath>     // for expm1f, pow, log1pf

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

// Function to clip values within a range
float clip(float n, float lower, float upper) {
    return std::max(lower, std::min(n, upper));
}

// First layer matrix multiplication
void m_multiply1(float train_new[64], float train[784]) {
    float res[64] = {0};
    for (int i = 0; i < 64; i++) {
        res[i] = 0;
        for (int j = 0; j < 784; j++) {
            res[i] += train[j] * weights1[i][j];
        }
    }
    for (int i = 0; i < 64; i++) {
        train_new[i] = res[i];
    }
}

// Second layer matrix multiplication
void m_multiply2(float train_new2[10], float train[64]) {
    float res[10] = {0};
    for (int i = 0; i < 10; i++) {
        res[i] = 0;
        for (int j = 0; j < 64; j++) {
            res[i] += train[j] * weights2[i][j];
        }
    }
    for (int i = 0; i < 10; i++) {
        train_new2[i] = res[i];
    }
}

// Update the voltage of neurons
void update_voltage(float* train, float* nodes, int n, float t, float* refractory_time) {
    float tau_rc = 0.02;
    float delta, nu, spike_mask, t_spike;

    for (int i = 0; i < n; i++) {
        refractory_time[i] -= t;
        nu = t - refractory_time[i];
        delta = clip(nu, 0, t);
        nodes[i] -= (train[i] - nodes[i]) * std::expm1f(-delta / tau_rc);

        if (nodes[i] > 1) train[i] = 1 / t;
        float eps = std::pow(10, -9);
        t_spike = t + tau_rc * std::log1pf(-1 * (nodes[i] - 1) / (train[i] - 1) + eps);
        if (nodes[i] < 0) nodes[i] = 0;

        refractory_time[i] = tau_ref + t_spike;
    }
}

// Main SNN LIF function
void snn_lif(axis_in *inp, axis_out *spike_sum) {
    int spikes[30][10] = {0};
    float train0[784] = {0};
    float train1[64] = {0};
    float train2[10] = {0};

    // Process each time step
    for (int t = 0; t < 30; t++) {
        for (int ind5 = 0; ind5 < 784; ind5++) {
            train0[ind5] = (*inp++).data;
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

        axis_out out;
        out.last = 0;
        out.data = tot;
        if (sn == 9) out.last = 1;
        *spike_sum++ = out;
    }
}
