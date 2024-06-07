#include <bits/stdc++.h>
using namespace std;
#include "snn_lif.hpp"

int main() {

    axis_in *inp;
    inp = (axis_in*) malloc(23520 * sizeof(axis_in));

    axis_out *spike_sum;
    spike_sum = (axis_out*) malloc(10 * sizeof(axis_out));

    if (!inp || !spike_sum) {
        cerr << "Memory allocation failed" << endl;
        return 1;
    }

    int epochs = 1, n_images = 10;
    for (int i = 0; i < epochs; i++) {
        for (int j = 1; j <= n_images; j++) {
            axis_in *temp = inp;
            for (int j = 0; j < 30; j++) {
                for (int k = 0; k < 784; k++) {
                    (*temp++).data = 0;
                }
            }
            temp = inp;

            string fname = "st" + to_string(j) + ".txt";
            cout << "Processing file: " << fname << endl;
            ifstream f(fname);
            if (f.is_open()) {
                for (int ind_a = 0; ind_a < 30; ind_a++) {
                    for (int ind_b = 0; ind_b < 784; ind_b++) {
                        if (!(f >> (*temp++).data)) {
                            cerr << "Error reading data from " << fname << " at position " << ind_a << ", " << ind_b << endl;
                            return 1;
                        }
                    }
                }
            } else {
                cerr << "Failed to open file " << fname << endl;
                return 1;
            }
            temp = inp;

            snn_lif(inp, spike_sum);

            // 输出结果
            axis_out *spike_temp = spike_sum;
            float max_val = (*spike_temp).data;
            int max_idx = 0;

            for (int i = 0; i < 10; i++) {
                float val = (*spike_temp).data;
                cout << val << " ";
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
                spike_temp++;
            }
            cout << endl;
            cout << "Predicted digit: " << max_idx << endl;

            // Reset spike_sum pointer for the next image
            spike_sum = (axis_out*) malloc(10 * sizeof(axis_out));
            if (!spike_sum) {
                cerr << "Memory allocation failed for spike_sum" << endl;
                return 1;
            }
        }
    }
    free(inp);
    free(spike_sum);
    return 0;
}
