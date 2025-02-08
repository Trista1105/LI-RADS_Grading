import csv

def write_to_csv(csvfile, list):
        writer = csv.writer(csvfile)
        writer.writerow(list)