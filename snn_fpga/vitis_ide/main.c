#include "xparameters.h"
#include "xsnn_lif.h"
#include "xil_printf.h"
#include "xuartps.h"
#include "xgpio.h"
#include "data.h"

#define ROWS 10
#define COLS 784
#define ARRAY_SIZE (ROWS * COLS)

// UART
#define UART_DEVICE_ID XPAR_XUARTPS_0_DEVICE_ID
#define LED_CHANNEL 1
#define GPIO_DEVICE_ID XPAR_GPIO_0_DEVICE_ID


XUartPs Uart_Ps;

int init_uart() {
    XUartPs_Config *Config;
    int Status;


    Config = XUartPs_LookupConfig(UART_DEVICE_ID);
    if (NULL == Config) {
        return XST_FAILURE;
    }

    Status = XUartPs_CfgInitialize(&Uart_Ps, Config, Config->BaseAddress);
    if (Status != XST_SUCCESS) {
        return XST_FAILURE;
    }


    XUartPs_SetBaudRate(&Uart_Ps, 115200);

    return XST_SUCCESS;
}

int main()
{
    XSnn_lif inout_test;
    int status;
    XGpio gpio;

    if (init_uart() != XST_SUCCESS) {
        xil_printf("UART Initialization failed\n");
        return XST_FAILURE;
    }
    status = XGpio_Initialize(&gpio, GPIO_DEVICE_ID);
    if (status != XST_SUCCESS) {
        return XST_FAILURE;
    }
    XGpio_SetDataDirection(&gpio, 1, 0x0);
    // 初始化IP核
    status = XSnn_lif_Initialize(&inout_test, XPAR_SNN_LIF_0_DEVICE_ID);
    if (status != XST_SUCCESS) {
        xil_printf("Initialization failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("Initialization!\r\n");

    // 分配最大值存储
    int max_value = 0;

//    for (int i = 0; i < ARRAY_SIZE; ++i) {
//    	xil_printf("in_array[%d] = %d\n",i,in_array[i]);
//    }

    XSnn_lif_Set_inp_offset(&inout_test, (u64)in_array);



    // 启动IP核
    XSnn_lif_Start(&inout_test);
    xil_printf("XSnn_lif_Start!\r\n");
    // 等待IP核完成
    while (!XSnn_lif_IsDone(&inout_test));
    xil_printf("XSnn_lif_IsDone!\r\n");
    // 读取最大值
    max_value = XSnn_lif_Get_max_index(&inout_test);

    // 打印最大值
    xil_printf("Max value: %d\r\n", max_value);
    xil_printf("Result expected : 3. \r\n");
    // 验证结果
    if (max_value == 3) {
        xil_printf("Test passed!\r\n");
    } else {
        xil_printf("Test failed!\r\n");
    }
    while(1){
    XGpio_DiscreteWrite(&gpio, LED_CHANNEL, max_value);
    }
    return 0;
}
