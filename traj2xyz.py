
import mdtraj
import sys

traj = sys.argv[1]
try:
    xyz = sys.argv[2]
except:
    xyz = 'output.xyz'
try:
    top = sys.argv[3]
except:
    top = None

if top is None:
    mol = mdtraj.load(traj)
else:
    mol = mdtraj.load(traj, top=top)

Element = [x.element.atomic_number for x in mol.top.atoms]

with open(xyz, 'w') as f:
    for idx, frame in enumerate(mol.xyz):
        f.write(f'{len(Element)}\nframe {idx}\n')
        for ele, coor in zip(Element, frame):
            f.write(f'{ele} {coor[0]*10:.3f} {coor[1]*10:.3f} {coor[2]*10:.3f}\n')
