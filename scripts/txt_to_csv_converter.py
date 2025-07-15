import csv

input_file = 'input.txt'   # Replace with your actual txt file name
output_file = 'output.csv' # Output CSV file name

with open(input_file, 'r', encoding='utf-8') as txtfile, \
     open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    
    txt_reader = csv.reader(txtfile, delimiter=';')  # Use semicolon as delimiter
    csv_writer = csv.writer(csvfile)

    for row in txt_reader:
        csv_writer.writerow(row)
