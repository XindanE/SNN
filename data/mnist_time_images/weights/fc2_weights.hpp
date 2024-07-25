const int8_t fc2_weights[10][64] = {
    {63, -13, 79, 91, -69, -25, -52, 22, -13, 28, -14, 32, -29, -2, -73, 3, -75, 67, 104, -82, 57, -1, 8, -5, 41, -67, 91, 71, 6, 89, -32, 74, 45, 81, -16, -45, 22, -9, 81, -59, -73, -3, 98, 37, 88, 83, -56, -41, 88, -86, -9, 20, -31, 87, 85, -26, -7, 16, 74, -16, -34, 53, -88, -68},
    {81, -14, -19, -43, 7, -62, 83, 127, 12, -74, -24, 72, -45, 89, 7, 65, 87, 98, 19, 63, -23, -65, -68, -66, -45, -55, -33, -51, -2, -92, 76, 0, 17, 95, 6, 90, 31, 76, -56, -62, 63, 98, -8, -50, -55, -3, 103, 43, -25, 76, 30, -14, 81, -65, -35, -23, 113, 68, -77, 51, -44, -25, 77, -89},
    {84, 33, 29, 40, -78, 55, 44, -128, 62, 57, 104, 46, 45, 46, 19, -31, -79, -31, 21, 55, 56, 59, -33, 97, -21, -37, 52, 89, -34, -32, -31, -27, -36, -49, -35, -76, 58, 26, 58, -52, -24, 47, -79, 4, 85, -58, 43, 74, -53, -53, 4, 55, 12, -65, 87, -30, 19, 99, -19, -74, 41, -62, 58, -26},
    {3, 50, -34, 76, 51, 9, -8, 12, 109, 65, 74, -53, -29, 52, 70, 31, 40, 8, -56, -61, 50, -58, -60, -61, -7, 68, 0, 52, -26, -60, -46, -72, 43, 1, 44, -27, -20, 10, 55, -36, 47, 93, -68, -45, -21, 92, 78, 39, 73, -57, 88, 20, -38, -10, 76, -31, 66, -44, -58, 98, 40, 6, -3, 70},
    {69, -63, 77, -57, 91, -89, 10, 18, -15, -11, 21, -54, -72, -59, -79, -35, 76, 35, -48, -45, 76, 67, 59, 70, -49, -29, -62, 17, 93, 35, 10, 84, 16, -15, 114, 104, -13, -75, 38, 51, 82, 92, -6, 2, -17, -53, -78, 97, -102, 34, 3, -79, -25, 22, -44, 68, 10, 3, 101, 44, 39, -26, 24, 88},
    {-31, 60, 48, -37, 93, -16, -64, 125, 96, 96, 48, 83, -73, 75, -46, 87, 73, -47, 30, -57, 77, -61, 71, -16, 8, -37, 82, -87, -56, 15, -41, 64, 70, 105, 69, -12, -21, 44, 80, 56, -82, -42, 18, 18, 55, 24, 14, -76, -38, -77, -52, -76, -45, 60, -29, 14, -116, -35, -4, 21, 79, 88, 79, 58},
    {-46, 53, 103, 86, 81, -53, -90, 69, -5, 90, -15, 75, 44, -16, -78, -5, 41, -6, 51, 65, -22, 24, 96, 82, -19, 28, -5, 4, 105, -6, 59, 91, -81, -31, -75, -45, 70, 64, -50, -24, -83, -111, 63, -73, 1, -2, -94, 46, -45, 15, -50, 33, 60, -81, 63, 88, -62, -57, -29, 8, -32, -11, 32, 43},
    {-62, 90, -66, -88, -2, 110, 8, 72, -89, -56, 56, -91, 58, -40, 26, 3, -55, 16, -12, -18, -40, 65, -58, 2, 90, -80, 100, -62, 6, -45, -3, -75, 24, 46, -33, 1, -47, 51, 52, 96, 14, 8, 78, 86, -24, -1, -31, 73, 40, 48, 104, -6, 64, 42, 79, 58, 44, -57, 91, 111, -57, -41, -49, 41},
    {101, -31, -61, -6, -10, -75, 57, -81, 10, 0, 86, 29, -14, 10, 55, 51, 40, -72, 33, 18, 98, 72, 63, 3, -91, 95, -3, -53, -23, 50, -79, 31, 104, -110, -83, -44, -67, 71, 29, 17, -8, -89, -70, -43, 11, 55, -45, -55, -45, -17, 37, 31, 64, 16, -47, 80, -53, 72, -76, -50, -4, 102, -4, 50},
    {-54, 69, -40, -34, 55, 44, 99, 55, 26, -21, -15, -40, -7, -85, 54, -63, 50, -10, 71, 20, -40, 78, 19, -62, -60, 93, -15, 92, -65, 37, 96, -20, -12, -13, 42, 48, 70, 27, -66, 28, 27, -5, 10, 97, 61, 27, -13, 35, -15, 47, 50, -47, -46, 58, -33, 84, 26, -77, 62, -81, 107, 73, -39, 72},
};
const float fc2_scale = 0.002044;
const int fc2_zero_point = 14;