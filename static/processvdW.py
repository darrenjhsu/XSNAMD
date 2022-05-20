
from vdW import *


print(f'float vdW[{len(vdW_table)}] = {{')
print('// H, C, N, O, S, and Fe are replaced with numbers from CRYSOL values (Svergun 1995)')
for idx, element in enumerate(vdW_table.keys()):
    print(f'{vdW_table[element]/100}', end='')
    if idx < len(vdW_table):
        print(',', end='')
    print(f' // {element}')
print('};')

