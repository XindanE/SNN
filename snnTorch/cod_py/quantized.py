import torch
import torch.nn as nn
import snntorch as snn

# 定义或加载你的模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 1000)
        self.lif1 = snn.Leaky(beta=0.95)
        self.fc2 = nn.Linear(1000, 10)
        self.lif2 = snn.Leaky(beta=0.95)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        mem1 = self.lif1.init_leaky().to(x.device)
        mem2 = self.lif2.init_leaky().to(x.device)
        spk2_rec, mem2_rec = [], []

        for step in range(25):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# 初始化模型
model = Net()

# 加载预训练模型
model.load_state_dict(torch.load("model.pth"))



# 准备量化
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# 校准模型（用部分数据）
model.train()
dummy_input = torch.randn(32, 28*28)
model(dummy_input)

# 转换量化模型
model.eval()
torch.quantization.convert(model, inplace=True)

# 保存量化后的模型
torch.save(model.state_dict(), "quantized_model.pth")