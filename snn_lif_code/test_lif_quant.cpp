#include <bits/stdc++.h>
using namespace std;
#include "snn_lif.hpp"

int main() {
    ap_uint<8> inp[784];
    ap_uint<16> spike_sum[10];

    int epochs = 1, n_images = 10;
    for (int i = 0; i < epochs; i++) {
        for (int j = 1; j <= n_images; j++) {
            fill(begin(inp), end(inp), 0);

            string fname = "st" + to_string(j) + ".txt";
            cout << "Processing file: " << fname << endl;
            ifstream f(fname);
            if (f.is_open()) {
                for (int ind_a = 0; ind_a < 30; ind_a++) {
                    for (int ind_b = 0; ind_b < 784; ind_b++) {
                        if (!(f >> inp[ind_b])) {
                            cerr << "Error reading data from " << fname << " at position " << ind_a << ", " << ind_b << endl;
                            return 1;
                        }
                    }
                }
                f.close();
            } else {
                cerr << "Failed to open file " << fname << endl;
                return 1;
            }

            snn_lif(inp, spike_sum);

            // 输出结果
            float max_val = spike_sum[0];
            int max_idx = 0;

            for (int i = 0; i < 10; i++) {
                float val = spike_sum[i];
                cout << val << " ";
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
            cout << endl;
            cout << "Predicted digit: " << max_idx << endl;
        }
    }
    return 0;
}
