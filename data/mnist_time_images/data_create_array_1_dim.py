# Function to read the data from the input file and convert it to a 1D list
def read_and_format_data_to_1d(input_file, time_steps=10, image_size=784):
    with open(input_file, 'r') as file:
        data = file.read()
    
    lines = data.split('\n')
    all_numbers = [int(num) for line in lines for num in line.split()]
    
    mnist_data = all_numbers[:time_steps * image_size]
    return mnist_data

# Function to save the 1D list to an output text file in the required C array format
def save_data_to_file_1d(data, output_file, array_name="mnist_data"):
    with open(output_file, 'w') as file:
        file.write("int mnist_data[10*784] = {\n")
        for i in range(0, len(data), 28):  # Write 16 numbers per line for better readability
            file.write("\t" + ', '.join(map(str, data[i:i + 28])) + ",\n")
        file.write("};\n")

# Define the input and output file paths
input_file = r"C:\Documents\infodoc\SNN\SNN_code\data\mnist\step_10\st3.txt"
output_file = r"C:\Documents\infodoc\SNN\SNN_code\data\mnist\step_10\st3_t1.txt"

# Read and format the data
mnist_data_1d = read_and_format_data_to_1d(input_file)

# Save the formatted data to the output file
save_data_to_file_1d(mnist_data_1d, output_file)