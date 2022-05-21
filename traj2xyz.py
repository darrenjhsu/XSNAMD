
import mdtraj
import sys
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--traj', type=str, help='File name of the pdb or dcd or xyz')
parser.add_argument('--top', type=str, default=None, help='Topology file for certain trajectories')
parser.add_argument('--out', type=str, default='output.axyz', help='Output name of the special xyz format')
parser.add_argument('--selection', type=str, default='protein', help='Molecular selection of the trajectory')

args = parser.parse_args()

if args.top is None:
    mol = mdtraj.load(args.traj)
else:
    mol = mdtraj.load(args.traj, top=args.top)

mol_sel = mol.top.select(args.selection)
mol = mol.atom_slice(mol_sel, inplace=True)

Element = [x.element.atomic_number for x in mol.top.atoms]

print(f'Trajectory: {args.traj}')
print(f'Topology: {args.top}')
print(f'Output: {args.out}')
print(f'Selection: {args.selection}')
print(f'Number of frames: {len(mol)}')

t0 = time.time()
t1 = time.time()

with open(args.out, 'w') as f:
    for idx, frame in enumerate(mol.xyz):
        t2 = time.time()
        if t2 - t1 > 5: 
            print(f'{t2 - t0} seconds, at frame {idx}')
            t1 = t2
        f.write(f'{len(Element)}\n{len(mol)} frames\n')
        for ele, coor in zip(Element, frame):
            f.write(f'{ele} {coor[0]*10:.3f} {coor[1]*10:.3f} {coor[2]*10:.3f}\n')
