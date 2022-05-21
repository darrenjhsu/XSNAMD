
import mdtraj
import sys
import argparse

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


with open(args.out, 'w') as f:
    for idx, frame in enumerate(mol.xyz):
        f.write(f'{len(Element)}\n{idx} frames\n')
        for ele, coor in zip(Element, frame):
            f.write(f'{ele} {coor[0]*10:.3f} {coor[1]*10:.3f} {coor[2]*10:.3f}\n')
