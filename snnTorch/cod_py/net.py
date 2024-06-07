import torch
import torch.nn as nn
import snntorch as snn

# 定义或加载你的模型
batch_size = 1
data_path = '/data/mnist'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CUDA Available:", torch.cuda.is_available())

# 网络架构参数
num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10

num_steps = 25
beta = 0.95

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # 在正确的设备上初始化隐藏状态
        x = x.view(x.size(0), -1)
        x = x.to(device)
        mem1 = self.lif1.init_leaky().to(device)
        mem2 = self.lif2.init_leaky().to(device)

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


def convert_onnx(model):
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True).to(device)
    torch.onnx.export(model,
                      dummy_input,
                      "model.onnx",
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    model = Net().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    print(model)
    convert_onnx(model)

    model.eval()
    # 创建虚拟输入数据
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    # 运行模型
    output = model(dummy_input)
    print(output)  # 输出形状应为 (num_steps, batch_size, num_outputs)

# # 加载模型
# model = Net()
# model.load_state_dict(torch.load("model.pth"))
# model.eval()
#
# # 创建一个虚拟输入张量
# dummy_input = torch.randn(1, 28 * 28)
#
# # 导出为 ONNX 格式
# torch.onnx.export(model, dummy_input, "model.onnx", input_names=['input'], output_names=['output'])