import csv

input_file = '../Data/electricityloaddiagrams20112014/LD2011_2014.txt'  
output_file = '../Data/power.csv'

with open(input_file, 'r', encoding='utf-8') as txtfile, \
     open(output_file, 'w', newline='', encoding='utf-8') as csvfile:

    txtfile = txtfile.read()
    txtfile = txtfile.replace(',', ';')


    txt_reader = csv.reader(txtfile, delimiter=';')  
    csv_writer = csv.writer(csvfile)

    for row in txt_reader:
        csv_writer.writerow(row)
