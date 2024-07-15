#include "Snn_lif.hpp"
//#include <algorithm> // for std::max and std::min
//#include <cmath>     // for expm1f, pow, log1pf

// Network architecture and parameters
const int T = 10;
const float pot_rest = 0.0f;
const float pot_thr = 1.0f;
const float tau_ref = 0.002f;
const float tau_rc = 0.02f;

// Nodes and refractory time initialization
float nodes0[784] = {0};
float nodes1[64] = {0};
float nodes2[10] = {0};

float refractory_time0[784] = {0.002f};
float refractory_time1[64] = {0.002f};
float refractory_time2[10] = {0.002f};

// Function to clip values within a range
float clip(float n, float lower, float upper) {
    return std::max(lower, std::min(n, upper));
}

// Matrix multiplication template function
template <int INPUT_SIZE, int OUTPUT_SIZE>
void matrix_multiply(int* output, const int* input, const int weights[OUTPUT_SIZE][INPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            output[i] += input[j] * weights[i][j];
        }
    }
}

// Update the voltage of neurons
void update_voltage(int* train, float* nodes, int n, float t, float* refractory_time, float scale) {
    for (int i = 0; i < n; i++) {
        refractory_time[i] -= t;
        float nu = t - refractory_time[i];
        float delta = clip(nu, 0, t);
        // 注意：这里将权重乘以了缩放因子scale
        nodes[i] -= (train[i] * scale - nodes[i]) * std::expm1f(-delta / tau_rc);

        if (nodes[i] > 1) train[i] = 1 / t;
        float eps = std::pow(10, -9);
        float t_spike = t + tau_rc * std::log1pf(-1 * (nodes[i] - 1) / (train[i] - 1) + eps);
        if (nodes[i] < 0) nodes[i] = 0;

        refractory_time[i] = tau_ref + t_spike;
    }
}

// Main SNN LIF function
void Snn_lif(int *inp, int *max_index) {

#pragma HLS INTERFACE m_axi port=inp  offset=slave depth=7840 bundle=inp max_widen_bitwidth=64
#pragma HLS INTERFACE s_axilite port=max_index bundle=max
#pragma HLS INTERFACE s_axilite port=return bundle=control



    int spikes[T][10] = {0};
    int train0[784];
    int train1[64];
    int train2[10];
    #pragma HLS ARRAY_PARTITION variable=train0 cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=train1 cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=train2 cyclic factor=8 dim=1

    // Initialize train0 with input data
    for (int t = 0; t < T; t++) {
        for (int ind5 = 0; ind5 < 784; ind5++) {
            train0[ind5] = inp[t * 784 + ind5];
        }

        // Update voltage for layer 0
        update_voltage(train0, nodes0, 784, dt[t], refractory_time0, 1.0f);
        // 输出train0的值

        // Reset neurons that have fired
        for (int sp = 0; sp < 784; sp++) {
            if (nodes0[sp] > pot_thr) nodes0[sp] = pot_rest;
        }
        //std::cout << "train0 values after voltage update at time step " << t << ":" << std::endl;
//        for (int i = 0; i < 784; i++) {
//            std::cout << nodes0[i] << " ";
//            if ((i + 1) % 28 == 0) // 每打印10个值换行，以便更好地阅读
//                std::cout << std::endl;
//        }
//        std::cout << std::endl;

        // Matrix multiplication for layer 1
        matrix_multiply<784, 64>(train1, train0, weights1);


        // Update voltage for layer 1
        update_voltage(train1, nodes1, 64, dt[t], refractory_time1, 952.5643219058368);

//        std::cout << "train1 values after voltage update at time step " << t << ":" << std::endl;
//        for (int i = 0; i < 64; i++) {
//            std::cout << nodes1[i] << " ";
//            if ((i + 1) % 10 == 0) // 每打印10个值换行，以便更好地阅读
//                std::cout << std::endl;
//        }
//        std::cout << std::endl;

        // Reset neurons that have fired
        for (int j = 0; j < 64; j++) {
            if (nodes1[j] > pot_thr) nodes1[j] = pot_rest;
        }

        // Matrix multiplication for layer 2
        matrix_multiply<64, 10>(train2, train1, weights2);

        // Update voltage for layer 2
        update_voltage(train2, nodes2, 10, dt[t], refractory_time2, 529.6544563520312);

//        std::cout << "train2 values after voltage update at time step " << t << ":" << std::endl;
//        for (int i = 0; i < 10; i++) {
//            std::cout << nodes2[i] << " ";
//            if ((i + 1) % 10 == 0) // 每打印10个值换行，以便更好地阅读
//                std::cout << std::endl;
//        }
//        std::cout << std::endl;

        // Record spikes
        for (int k = 0; k < 10; k++) {
            if (nodes2[k] > 1e+7) {
                nodes2[k] = pot_rest;
                if (k < 10 && t < T) { // 添加边界检查
                    spikes[t][k] = 1;
                }
            } else {
                if (k < 10 && t < T) { // 添加边界检查
                    spikes[t][k] = 0;
                }
            }
        }
    }

    // Sum spikes over all time steps and find the index of the maximum spike value
    int spike_sum[10] = {0};
    int max_val = 0;
    int max_idx = 0;
    for (int sn = 0; sn < 10; sn++) {
        int tot = 0;
        for (int sm = 0; sm < T; sm++) {
            tot += spikes[sm][sn];
        }
        spike_sum[sn] = tot;

        if (tot > max_val) {
            max_val = tot;
            max_idx = sn;
        }
    }

    *max_index = max_idx;
}
