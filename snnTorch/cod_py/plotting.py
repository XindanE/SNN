import matplotlib.pyplot as plt
import numpy as np
from snntorch import spikeplot as splt


# 定义用于绘图的函数
def plot_mem(u_trace, title):
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.plot(u_trace)  # 绘制 u_trace
    plt.title(title)  # 设置图形标题
    plt.xlabel("Time Step")  # 设置x轴标签
    plt.ylabel("Membrane Potential U")  # 设置y轴标签
    plt.xlim(0, len(u_trace)-1)  # 显式设置x轴范围，从0开始到u_trace的长度减1
    plt.show()  # 显示图形


def plot_step_current_response(cur_in, mem_rec, x):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # 创建一个包含两个子图的图形，共享x轴

    # 绘制输入电流
    axs[0].plot(cur_in.numpy(), color='orange')  # 假设cur_in是张量，需要转为numpy数组
    axs[0].set_title('Input Current (I_in)')
    axs[0].set_ylabel('Input Current (I_in)')
    axs[0].axvline(x=x, color='gray', linestyle='--')  # 假设在第10个时间步电流开始，添加垂直线

    # 绘制膜电位
    axs[1].plot(mem_rec.numpy(), color='blue')  # 假设mem_rec是张量，需要转为numpy数组
    axs[1].set_title("Membrane Potential (U_mem)")
    axs[1].set_ylabel('Membrane Potential (U_mem)')
    axs[1].set_xlabel('Time step')
    axs[1].axvline(x=x, color='gray', linestyle='--')  # 在相同位置添加垂直线

    # 添加总标题并显示图形
    plt.suptitle("Lapicque's Neuron Model With Step Input")
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # 调整布局以给总标题留出空间
    plt.show()


def plot_current_pulse_response(current_input, membrane_response, title, vline1, vline2=None, ylim_max1=None):
    # 设置图表大小和共享X轴
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 绘制输入电流
    axs[0].plot(current_input.numpy(), color='orange', linewidth=2)
    axs[0].set_title('Input Current (I_in)')
    axs[0].set_ylabel('Current (A)')
    axs[0].axvline(x=vline1, color='gray', linestyle='--')  # 标记电流输入的第一个关键时间点
    if vline2:
        axs[0].axvline(x=vline2, color='gray', linestyle='--')  # 如果提供了第二个时间点，则标记
    if ylim_max1 is not None:
        axs[0].set_ylim(0, ylim_max1)  # 设置y轴的范围，如果提供了最大值

    # 绘制膜电位
    axs[1].plot(membrane_response.numpy(), color='blue', linewidth=2)
    axs[1].set_title("Membrane Potential (U_mem)")
    axs[1].set_ylabel('Potential (V)')
    axs[1].set_xlabel('Time step')
    axs[1].axvline(x=vline1, color='gray', linestyle='--')  # 标记电流输入的第一个关键时间点
    if vline2:
        axs[1].axvline(x=vline2, color='gray', linestyle='--')  # 如果提供了第二个时间点，则标记

    # 添加总标题并显示图形
    plt.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def compare_plots(cur_in1, cur_in2, cur_in3, mem_rec1, mem_rec2, mem_rec3, start1, start2, start3, end, title):
    # 设置图表大小和共享X轴
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 设置颜色列表，方便区分不同的输入
    colors = ['orange', 'green', 'blue']

    # 绘制三组输入电流
    for cur_in, color in zip([cur_in1, cur_in2, cur_in3], colors):
        axs[0].plot(cur_in.numpy(), color=color, linewidth=2)
    axs[0].set_title('Input Current (I_in)')
    axs[0].set_ylabel('Current (A)')
    for start in [start1, start2, start3]:  # 标记所有起始点
        axs[0].axvline(x=start, color='gray', linestyle='--')
    axs[0].axvline(x=end, color='gray', linestyle='--')  # 标记结束点

    # 绘制三组膜电位响应
    for mem_rec, color in zip([mem_rec1, mem_rec2, mem_rec3], colors):
        axs[1].plot(mem_rec.numpy(), color=color, linewidth=2)
    axs[1].set_title("Membrane Potential (U_mem)")
    axs[1].set_ylabel('Potential (V)')
    axs[1].set_xlabel('Time step')
    for start in [start1, start2, start3]:  # 标记所有起始点
        axs[1].axvline(x=start, color='gray', linestyle='--')
    axs[1].axvline(x=end, color='gray', linestyle='--')  # 标记结束点

    # 添加总标题并显示图形
    plt.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_cur_mem_spk(current_input, membrane_potential, spikes, thr_line, title, ylim_max1=None, ylim_max2=None, vline=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 3, 1]}, sharex=True)  # 修改子图的高度比例

    # 绘制输入电流
    axs[0].plot(current_input.numpy(), color='orange', linewidth=2)
    axs[0].set_title('Input Current (I_in)')
    axs[0].set_ylabel('Current (A)')
    if vline :
        axs[0].axvline(x=vline, color='gray', linestyle='--')
    axs[0].set_ylim(0, ylim_max1)

    # 绘制膜电位
    axs[1].plot(membrane_potential.numpy(), color='blue', linewidth=2)
    axs[1].axhline(y=thr_line, color='red', linestyle='--')
    axs[1].set_title("Membrane Potential (U_mem)")
    axs[1].set_ylabel('Potential (V)')
    if vline:
        axs[1].axvline(x=vline, color='gray', linestyle='--')
    axs[1].set_ylim(0, ylim_max2)

    # 绘制尖峰发放图
    axs[2].eventplot(np.where(spikes.numpy().ravel())[0], orientation='horizontal', colors='black', linelengths=0.8)  # 增加线条长度
    axs[2].set_title("Output Spikes")
    axs[2].set_yticks([])
    axs[2].set_xlabel('Time step')
    if vline:
        axs[2].axvline(x=vline, color='gray', linestyle='--')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_spk_mem_spk(spk_in, mem_rec, spk_rec, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, gridspec_kw = {'height_ratios': [0.4, 1, 0.4]})

    # Plot input current
    splt.raster(spk_in, ax[0], s=400, c="black", marker="|")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)
    ax[0].set_yticks([])

    # Plot membrane potential
    ax[1].plot(mem_rec.detach())
    ax[1].set_ylim([0, 0.6])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[1].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk_rec, ax[2], s=400, c="black", marker="|")
    ax[2].set_yticks([])
    ax[2].set_ylabel("Output Spikes")

    plt.show()


def plot_reset_comparison(spike_input, mem_rec1, spk_rec1, mem_rec2, spk_rec2):
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))  # 3行2列的图

    # 绘制左侧的尖峰输入、膜电位和尖峰输出
    axs[0, 0].eventplot(np.where(spike_input.numpy().ravel())[0], orientation='horizontal', colors='black', linelengths=1)
    axs[0, 0].set_title("Input Spikes")
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xticks([])

    axs[1, 0].plot(mem_rec1.numpy(), color='blue')
    axs[1, 0].set_title("Membrane Potential - No Reset")
    axs[1, 0].set_ylabel("Potential (V)")
    axs[1, 0].set_xticks([])

    axs[2, 0].eventplot(np.where(spk_rec1.numpy().ravel())[0], orientation='horizontal', colors='black', linelengths=1)
    axs[2, 0].set_title("Output Spikes - No Reset")
    axs[2, 0].set_yticks([])
    axs[2, 0].set_xlabel("Time step")

    # 绘制右侧的尖峰输入、膜电位和尖峰输出
    axs[0, 1].eventplot(np.where(spike_input.numpy().ravel())[0], orientation='horizontal', colors='black', linelengths=1)
    axs[0, 1].set_title("Input Spikes")
    axs[0, 1].set_yticks([])
    axs[0, 1].set_xticks([])

    axs[1, 1].plot(mem_rec2.numpy(), color='blue')
    axs[1, 1].set_title("Membrane Potential - Reset to Zero")
    axs[1, 1].set_ylabel("Potential (V)")
    axs[1, 1].set_xticks([])

    axs[2, 1].eventplot(np.where(spk_rec2.numpy().ravel())[0], orientation='horizontal', colors='black', linelengths=1)
    axs[2, 1].set_title("Output Spikes - Reset to Zero")
    axs[2, 1].set_yticks([])
    axs[2, 1].set_xlabel("Time step")

    plt.tight_layout()
    plt.show()


def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8, 7), sharex=True, gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input spikes
    splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    # Plot hidden layer spikes
    splt.raster(spk1_rec.reshape(200, -1), ax[1], s = 0.05, c="black")
    ax[1].set_ylabel("Hidden Layer")

    # Plot output spikes
    splt.raster(spk2_rec.reshape(200, -1), ax[2], c="black", marker="|")
    ax[2].set_ylabel("Output Spikes")
    ax[2].set_ylim([0, 10])

    plt.show()


def plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, title):
    # Generate Plots
    fig, ax = plt.subplots(4, figsize=(8,7), sharex=True, gridspec_kw = {'height_ratios': [0.4, 1, 1, 0.4]})

    # Plot input current
    splt.raster(spk_in, ax[0], s=400, c="black", marker="|")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title("Synaptic Conductance-based Neuron Model With Input Spikes")
    ax[0].set_yticks([])

    # Plot membrane potential
    ax[1].plot(syn_rec.detach().numpy())
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel("Synaptic Current ($I_{syn}$)")
    plt.xlabel("Time step")

    # Plot membrane potential
    ax[2].plot(mem_rec.detach().numpy())
    ax[2].set_ylim([0, 1.5])
    ax[2].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[2].axhline(y=1, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk_rec, ax[3], s=400, c="black", marker="|")
    plt.ylabel("Output spikes")
    ax[3].set_yticks([])

    plt.show()


