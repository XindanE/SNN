# Function to read the data from the input file and convert it to a 2D list
def read_and_format_data(input_file, time_steps=10, image_size=784):
    with open(input_file, 'r') as file:
        data = file.read()
    
    lines = data.split('\n')
    all_numbers = [int(num) for line in lines for num in line.split()]
    
    mnist_data = [all_numbers[i:i + image_size] for i in range(0, len(all_numbers), image_size)][:time_steps]
    return mnist_data

# Function to save the 2D list to an output text file in the required C array format
def save_data_to_file(data, output_file):
    with open(output_file, 'w') as file:
        file.write("int mnist_data[10][784] = {\n")
        for row in data:
            file.write("\t{" + ', '.join(map(str, row)) + "},\n")
        file.write("};\n")

# Define the input and output file paths
input_file = r"C:\Documents\infodoc\SNN\SNN_code\data\mnist\step_10\st1.txt"
output_file = r"C:\Documents\infodoc\SNN\SNN_code\data\mnist\step_10\st1_t.txt"

# Read and format the data
mnist_data = read_and_format_data(input_file)

# Save the formatted data to the output file
save_data_to_file(mnist_data, output_file)