import os
import csv
'''
Splits large csv file into training and testing split csv files
'''
with open('alpha.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        x = row[0].split(',')
        for file in os.listdir('/home/fredd/Documents/University/CI601 - The Computing Project/WLASL/start_kit/videos'):
            if x[0] + '.mp4' == file:
                print('present')
                if x[1] == 'test':
                    f = open('alphatest.csv', 'a')
                    writer = csv.writer(f)
                    writer.writerow([x[0],x[2]])
                if x[1] == 'train' or "val":
                    f = open('alphatrain.csv', 'a')
                    writer = csv.writer(f)
                    writer.writerow([x[0],x[2]])
