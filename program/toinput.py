'''
Converts space or comma separated files into a working 
input file. First command argument is the name of the file, and the
second the number of wanted output points.

Johan Nordstrom 2016
'''

import sys

if len(sys.argv) < 3:
    print 'No input file or number of points'
    sys.exit()

name = sys.argv[1]
num = int(sys.argv[2])
input_file = open(name, 'r')
output_file = open('infloa.in', 'w')
i = 0
for s in input_file:
    if i >= num : break
    nums = s.strip().replace(',', ' ').split()
    floats = [float(x) for x in nums]
    string = " ".join(str(x) for x in floats)
    output_file.write(string + '\n')
    i = i + 1
