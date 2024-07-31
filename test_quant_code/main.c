#include "xparameters.h"
#include "xsnn_quant.h"
#include "xil_printf.h"
#include "xuartps.h"
#include "xgpio.h"
#include "data.h"

#define ROWS 10
#define COLS 784
#define ARRAY_SIZE (ROWS * COLS)

// Define constants for UART and GPIO setup
#define UART_DEVICE_ID XPAR_XUARTPS_0_DEVICE_ID
#define LED_CHANNEL 1
#define GPIO_DEVICE_ID XPAR_GPIO_0_DEVICE_ID

XUartPs Uart_Ps;

// Initialize UART communication
int init_uart() {
    XUartPs_Config *Config;
    int Status;

    // Look up the configuration based on the device ID
    Config = XUartPs_LookupConfig(UART_DEVICE_ID);
    if (NULL == Config) {
        return XST_FAILURE;
    }

    Status = XUartPs_CfgInitialize(&Uart_Ps, Config, Config->BaseAddress);
    if (Status != XST_SUCCESS) {
        return XST_FAILURE;
    }

    // Set the baud rate for UART
    XUartPs_SetBaudRate(&Uart_Ps, 115200);

    return XST_SUCCESS;
}

void print_float(float f) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%f", f);
    xil_printf("%s", buffer);
}

int main()
{
    XSnn_quant inout_test;
    int status;
    //XGpio gpio;
    int correct=0;

    // Initialize UART and check for errors
    if (init_uart() != XST_SUCCESS) {
        xil_printf("UART Initialization failed\n");
        return XST_FAILURE;
    }
    // Initialize XSnn_quant
    status = XSnn_quant_Initialize(&inout_test, XPAR_SNN_QUANT_0_DEVICE_ID);
    if (status != XST_SUCCESS) {
        xil_printf("SNN Quant Initialization failed\n");
        return XST_FAILURE;
    }


/*    XSnn_quant_Initialize(&inout_test, XPAR_SNN_QUANT_0_DEVICE_ID);
                XSnn_quant_Set_inp_offset(&inout_test, (u64)samples[3].data);

                XSnn_quant_Start(&inout_test);

                while (!XSnn_quant_IsDone(&inout_test));

                int predicted_label = XSnn_quant_Get_max_index(&inout_test);
                if (predicted_label == samples[3].label) {
                    xil_printf("Sample %d: Test passed! Predicted = %d, Actual = %d\n", 3, predicted_label, samples[3].label);
                } else {
                    xil_printf("Sample %d: Test failed! Predicted = %d, Actual = %d\n", 3, predicted_label, samples[3].label);
                }*/

    for (int i = 0; i < 10; i++) {

        XSnn_quant_DisableAutoRestart(&inout_test);


        XSnn_quant_Set_inp_offset(&inout_test, (u64)samples[i].data);


        XSnn_quant_Start(&inout_test);


        while (!XSnn_quant_IsDone(&inout_test));

        XSnn_quant_EnableAutoRestart(&inout_test);


        int predicted_label = XSnn_quant_Get_max_index(&inout_test);
        if (predicted_label == samples[i].label) {
            xil_printf("Sample %d: Test passed! Predicted = %d, Actual = %d\n", i, predicted_label, samples[i].label);
            correct++;
        } else {
            xil_printf("Sample %d: Test failed! Predicted = %d, Actual = %d\n", i, predicted_label, samples[i].label);
        }
    }
    xil_printf("Precision:\n");
    float rate = correct/10.0;
    print_float(rate);

   /* // Initialize GPIO for output
    status = XGpio_Initialize(&gpio, GPIO_DEVICE_ID);
    if (status != XST_SUCCESS) {
        return XST_FAILURE;
    }
    XGpio_SetDataDirection(&gpio, 1, 0x0);
    // Initialize the IP core
    status = XSnn_quant_Initialize(&inout_test, XPAR_SNN_QUANT_0_DEVICE_ID);
    if (status != XST_SUCCESS) {
        xil_printf("Initialization failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("Initialization!\r\n");

    // Set the offset for input data
    XSnn_quant_Set_inp_offset(&inout_test, (u64)in_array);

    // Start the IP core processing
    XSnn_quant_Start(&inout_test);
    xil_printf("XSnn_quant_Start!\r\n");
    // Wait for the IP core to complete processing
    while (!XSnn_quant_IsDone(&inout_test));
    xil_printf("XSnn_quant_IsDone!\r\n");

    // Retrieve the maximum value result from the IP core
    int max_value = XSnn_quant_Get_max_index(&inout_test);

    // Print the maximum value obtained
    xil_printf("Max value: %d\r\n", max_value);
    xil_printf("Result expected : 3. \r\n");
    // Validate the result against expected outcome
    if (max_value == 3) {
        xil_printf("Test passed!\r\n");
    } else {
        xil_printf("Test failed!\r\n");
    }
    // Continuously write the max value to the GPIO for LED display
    while(1){
        XGpio_DiscreteWrite(&gpio, LED_CHANNEL, max_value);
    }*/
    return 0;
}
