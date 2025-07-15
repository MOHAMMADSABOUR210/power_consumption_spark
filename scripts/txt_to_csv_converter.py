import csv

input_file = '../Data/electricityloaddiagrams20112014/LD2011_2014.txt'   # Replace with your actual txt file name
output_file = '../Data/power.csv' # Output CSV file name

with open(input_file, 'r', encoding='utf-8') as txtfile, \
     open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    
    txt_reader = csv.reader(txtfile, delimiter=';')  # Use semicolon as delimiter
    csv_writer = csv.writer(csvfile)

    for row in txt_reader:
        csv_writer.writerow(row)
