import torch
import torch.nn as nn
import snntorch as snn
from snntorch.functional import quant
from torchvision import datasets, transforms

# 设置参数
batch_size = 1  # ONNX 导出时 batch_size 设置为 1
num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10
num_steps = 25
beta = 0.95
threshold = 1
num_bits = 8
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 定义量化感知训练的 SNN 模型
class QuantizedSNN(nn.Module):
    def __init__(self):
        super(QuantizedSNN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.q_lif1 = quant.state_quant(num_bits=num_bits, uniform=True, threshold=threshold)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, state_quant=self.q_lif1)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.q_lif2 = quant.state_quant(num_bits=num_bits, uniform=True, threshold=threshold)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, state_quant=self.q_lif2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = x.to(device)
        mem1 = self.lif1.init_leaky().to(device)
        mem2 = self.lif2.init_leaky().to(device)
        spk2_rec = []
        mem2_rec = []
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1.to(device), mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2.to(device), mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


def convert_onnx(model):
    model.eval()
    dummy_input = torch.randn(1, num_inputs, requires_grad=True).to(device)
    torch.onnx.export(model,
                      dummy_input,
                      "model_quant_snn.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    model = QuantizedSNN().to(device)
    model.load_state_dict(torch.load("quant_model.pth", map_location=device))
    convert_onnx(model)
#
# # 加载训练好的模型
# model = QuantizedSNN().to(device)
# model.load_state_dict(torch.load('./model_test.pth', map_location=device))
# model.eval()
#
# # 创建一个随机输入张量
# dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)
#
# # 导出为 ONNX 模型
# torch.onnx.export(model.to(device), dummy_input.to(device), "quantized_snn.onnx", export_params=True, opset_version=11,
#                   input_names=['input'], output_names=['output'])
