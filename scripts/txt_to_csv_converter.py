import csv

input_file = '../Data/electricityloaddiagrams20112014/LD2011_2014.txt'  
# output_file = '../Data/power.csv'

# input_file = 'input.txt'
output_file = '../Data/output.txt'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        updated_line = line.replace(',', ';')
        outfile.write(updated_line)


# with open(input_file, 'r', encoding='utf-8') as txtfile, \
#      open(output_file, 'w', newline='', encoding='utf-8') as csvfile:

#     txtfile = txtfile.read()
#     txtfile = txtfile.replace(',', ';')


#     txt_reader = csv.reader(txtfile, delimiter=';')  
#     csv_writer = csv.writer(csvfile)

#     for row in txt_reader:
#         csv_writer.writerow(row)
