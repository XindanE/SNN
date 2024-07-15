def read_weights(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        weights_str = ''.join(lines)
        weights_str = weights_str.split('= {')[1].rsplit('};', 1)[0]
        weights_str = weights_str.replace('\n', '').replace('{', '').replace('}', '')
        weights_list = weights_str.split(',')
        weights = [float(weight) for weight in weights_list if weight.strip()]
    return weights

def quantize_weights(weights, scale):
    quantized_weights = [round(weight * scale) for weight in weights]
    # 将量化的权重限制在8位整数范围内（-128到127）
    quantized_weights = [max(min(w, 127), -128) for w in quantized_weights]
    return quantized_weights

def save_quantized_weights(file_path, quantized_weights, shape, array_name):
    with open(file_path, 'w') as file:
        file.write(f'{array_name} = {{\n')
        for i in range(shape[0]):
            line = quantized_weights[i * shape[1]:(i + 1) * shape[1]]
            file.write('{' + ','.join(map(str, line)) + '},\n')
        file.write('};\n')

def reshape_weights(weights, shape):
    reshaped = []
    for i in range(shape[0]):
        reshaped.append(weights[i * shape[1]:(i + 1) * shape[1]])
    return reshaped

def main():
    # 读取并量化weights1
    weights1 = read_weights('weights1.txt')
    max_abs_weight1 = max(abs(weight) for weight in weights1)
    scale1 = 127 / max_abs_weight1
    print("scale1=%f",scale1)
    weights1_quant = quantize_weights(weights1, scale1)
    weights1_quant_reshaped = reshape_weights(weights1_quant, (64, 784))
    save_quantized_weights('weights1_quant.txt', [w for row in weights1_quant_reshaped for w in row], (64, 784), 'weights1_quant')

    # 读取并量化weights2
    weights2 = read_weights('weights2.txt')
    max_abs_weight2 = max(abs(weight) for weight in weights2)
    scale2 = 127 / max_abs_weight2
    print("scale2=%f",scale2)
    weights2_quant = quantize_weights(weights2, scale2)
    weights2_quant_reshaped = reshape_weights(weights2_quant, (10, 64))
    save_quantized_weights('weights2_quant.txt', [w for row in weights2_quant_reshaped for w in row], (10, 64), 'weights2_quant')

if __name__ == '__main__':
    main()
