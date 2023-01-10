from ase.calculators.lj import LennardJones
from ase.io import read, write
from ase.optimize import BFGS
from ase.build.bulk import bulk
import numpy as np
from PathIntegrationTest import PathIntegrationTest


def main():
    atoms = read('../test/poslow_000.001.xyz')
    #atoms.set_positions(atoms.get_positions() / Bohr)
    atoms.calc = LennardJones(rc = 50, smooth=True)
    write('../test/in.xyz', atoms)

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(e, np.linalg.norm(f))

    opt = BFGS(atoms)
    opt.run(steps=30)

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(e, np.linalg.norm(f))

    atoms.set_positions(atoms.get_positions() + np.random.randn(*atoms.get_positions().shape) * 0.2)

    write('../test/out.xyz', atoms)

    pathIntTest = PathIntegrationTest(atoms, 0.5, 100, shape='pentagram')
    pathIntTest.integrate()
    pathIntTest.plot_energy_along_path()
    pathIntTest.plot_error_along_path()
    pathIntTest.plot_pentagram_energy()
    pathIntTest.plot_pentagram_error()

def main_periodic():
    atoms = bulk('Ar', 'bcc', 1.233, orthorhombic=True).repeat((2, 2, 2))
    atoms.calc = LennardJones(rc = 10, smooth=True)
    write('../test/in_periodic.extxyz', atoms)
    write('../test/in_periodic.ascii', atoms)

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress()
    print(e, np.linalg.norm(f), np.linalg.norm(s))

if __name__ == '__main__':
    main()