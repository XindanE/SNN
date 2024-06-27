def generate_c_array_from_txt(file_path, output_path):
    # 读取文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 处理数据生成 C 语言数组格式
    with open(output_path, 'w') as out_file:
        out_file.write("uint8_t mnist_data[30][784] = {\n")
        for line in lines:
            numbers = line.strip().split()
            if len(numbers) != 784:
                raise ValueError("Each line must contain 784 numbers")
            array_line = "{" + ", ".join(numbers) + "},\n"
            out_file.write(array_line)
        out_file.write("};\n")

# 调用函数，这里需要确保路径用引号引起来，并且路径是正确的
generate_c_array_from_txt('/home/xindan/snn_project/files/st1.txt', 'output_c_code.c')

