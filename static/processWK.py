
with open(f'f0_WaasKirf.dat', 'r') as f:
    cont = f.readlines()


this_atomic_number = None
num_comp = 0
values = []
elements = []
for idx, line in enumerate(cont):
    if '#S' in line:
        if line.split()[1] != this_atomic_number: # New element
            this_atomic_number = line.split()[1]
            values.append(cont[idx+3])
            elements.append(line.split()[2])
            num_comp += 11


print(f'float WK[{num_comp}] = {{')
for idx, line in enumerate(values):
    print(f'// {elements[idx]}')
    for idy, ele in enumerate(line.strip('\n').split()):
        if idy < len(line.strip('\n').split())-1:
            print(f'{ele}, ', end='')
        else:
            print(f'{ele}', end='')
    if idx < len(values)-1:
        print(',')
print('};\n\n')
    
    
