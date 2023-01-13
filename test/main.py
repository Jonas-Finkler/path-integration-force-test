from ase.calculators.lj import LennardJones
from ase.io import read, write
from ase.optimize import BFGS
from ase.build.bulk import bulk
import numpy as np
from pathIntegrationForceTest.PathIntegrationTest import PathIntegrationTest


def main():
    # This is an example for a system with free boundary conditions.
    # Using the global ground state of the LJ_38 cluster
    atoms = read('../test/poslow_000.001.xyz')
    atoms.calc = LennardJones(rc = 50, smooth=True)
    write('../test/in.xyz', atoms)

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(e, np.linalg.norm(f))

    #opt = BFGS(atoms)
    #opt.run(steps=30)
    #e = atoms.get_potential_energy()
    #f = atoms.get_forces()
    #print(e, np.linalg.norm(f))

    # Setup: The pentagram is most fun
    pathIntTest = PathIntegrationTest(atoms, 0.2, 100, shape='pentagram', startingPointIsOnCircle=False)
    # run the integration
    pathIntTest.integrate()
    # plot the energy and error
    pathIntTest.plot_energy_along_path()
    pathIntTest.plot_error_along_path()
    # with the pentagram we can make these nice star shaped plots if we circle a local minimum
    pathIntTest.plot_pentagram_energy()
    pathIntTest.plot_pentagram_error()

    pathIntTest.write_trajectory('../test/traj.extxyz')

    # if you don't want to use matplotlib you can write it to a file and plot it yourself
    pathIntTest.write_to_file('../test/path_int_test.txt')

    # displace the atoms away from the local minimum
    atoms.set_positions(atoms.get_positions() + np.random.randn(*atoms.get_positions().shape) * 0.2)
    # the circle test is maybe the most usefule one
    pathIntTest = PathIntegrationTest(atoms, 0.2, 100, shape='circle', startingPointIsOnCircle=False)
    max_error, energy_range = pathIntTest.integrate()
    # we can use these values to compute a measure for the accuracy
    print(max_error / energy_range)
    pathIntTest.plot_energy_along_path()
    pathIntTest.plot_error_along_path()

def main_periodic():
    atoms = bulk('Ar', 'bcc', 1.233, orthorhombic=True)#.repeat((2, 2, 2))
    atoms.calc = LennardJones(rc = 10, smooth=True)
    write('../test/in_periodic.extxyz', atoms)
    write('../test/in_periodic.ascii', atoms)

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress()
    print(e, np.linalg.norm(f), np.linalg.norm(s))

    # It also works with periodic systems
    pathIntTest = PathIntegrationTest(atoms, 0.01, 20, shape='circle', check_stress=False)
    pathIntTest.integrate()
    pathIntTest.plot_energy_along_path()
    pathIntTest.plot_error_along_path()

    # We can also do the same test for the stress calculations
    pathIntTest = PathIntegrationTest(atoms, 0.01, 20, shape='circle', check_stress=True)
    pathIntTest.integrate()
    pathIntTest.plot_energy_along_path()
    pathIntTest.plot_error_along_path()

if __name__ == '__main__':
    main()
    main_periodic()