from ase import Atoms
from ase.io import write
import numpy as np

class PathIntegrationTest:
    def __init__(self, center: Atoms, radius: float, nsteps: int, check_stress=False, shape='circle', verbose=True, randomTrajectory = False, startingPointIsOnCircle = True):
        """
        Integrate forces along path to check if derivatives of energy are correctly implemented
        :param center: Geoemtry at the center of the circle
        :param radius: Radius of the circle to be integrated
        :param nsteps: Number of integration steps
        :param check_stress: If true the lattice vectors are integrated instead of the atomic positions
        :param shape: Can be 'circle', 'line' or 'pentagram'
        :param verbose: Print output to stdout if true
        :param randomTrajectory: if false, random seed is set before trajectory is generated
        :param startingPointIsOnCircle: if true, starting point is on circles
        """
        self.center = center
        self.radius = radius
        self.nsteps = nsteps
        self.verbose = verbose
        self.check_stress = check_stress
        self.startingPointIsOnCircle = startingPointIsOnCircle

        if not randomTrajectory:
            np.random.seed(7342863)

        if self.check_stress:
            self.center_pos = self.center.get_cell(True)
        else:
            self.center_pos = self.center.get_positions()
        nats = self.center_pos.shape[0]
        # todo: in case of lattice: choose d1, d2 such that volume is not changed.
        # todo: remove translation and rotation?
        d1 = np.random.randn(nats, 3)
        d2 = np.random.randn(nats, 3)
        if not self.check_stress:
            d1 = self.__remove_translation(d1)
            d2 = self.__remove_translation(d2)
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)
        d2 = d2 - d1 * np.sum(d2 * d1)
        d2 = d2 / np.linalg.norm(d2)
        assert np.sum(d1 * d2) < 1.e-14, 'd1 is not orthogonal to d2'
        self.d1, self.d2 = d1, d2

        self.energies = []
        self.integrated_energies = []
        self.energy_derivatives = []
        self.positions = []

        # todo: ellipse
        self.shape = shape
        if self.shape == 'circle':
            self.positions = self.__circle(self.nsteps)
            # close the circle
            self.positions.append(self.positions[0])
        elif self.shape == 'line':
            self.positions = self.__line(self.nsteps)
        elif self.shape == 'pentagram':
            self.positions = self.__pentagram(self.nsteps)
            # close the circle
            self.positions.append(self.positions[0])


    @staticmethod
    def __remove_translation(x: np.ndarray):
        '''
        Projects out translational motion from x
        :param x: input vector
        :return: x with all three translational dimensions projected out
        '''
        y = x.copy()
        for i in range(3): # x, y, z
            tv = np.zeros(y.shape)
            tv[:,i] = 1.
            y = y - tv * np.sum(y * tv) / np.sum(tv**2)
        return y




    def __get_pentagram_nsteps(self, nsteps):
        corners = self.__circle(5)
        l_circ = 2. * np.pi * self.radius
        l_line = np.linalg.norm(corners[2] - corners[0])
        nsteps_line = int((5 * l_line + l_circ) / l_line)
        nsteps_circ = nsteps - 5 * nsteps_line
        return nsteps_circ, nsteps_line

    def __pentagram(self, nsteps):
        corners = self.__circle(5)
        nsteps_circ, nsteps_line = self.__get_pentagram_nsteps(nsteps)
        #print(nsteps_circ, nsteps_line, nsteps_circ + 5 * nsteps_line, nsteps)
        positions = self.__circle(nsteps_circ)
        for i in range(5):
            positions.extend(self.__line(nsteps_line, corners[(i * 2) % 5], corners[(i * 2 + 2) % 5]))
        return positions


    def __circle(self, nsteps):
        angles = np.linspace(0, 2 * np.pi, nsteps, endpoint=False)
        positions = []
        for i, angle in enumerate(angles):
            if i == 0:
                if self.startingPointIsOnCircle:
                    displacement = (self.d1 * np.cos(angle) + self.d2 * np.sin(angle)) * self.radius
                else:
                    displacement = 0.0
            positions.append(self.center_pos + (self.d1 * np.cos(angle) + self.d2 * np.sin(angle)) * self.radius - displacement)
        return positions


    def __line(self, nsteps, posA=None, posB=None):
        if posA is None and posB is None:
            posA = self.center_pos
            d = np.random.randn(*posA.shape)
            d = d / np.linalg.norm(d) * self.radius
            posB = posA + d
        lam = np.linspace(0., 1., nsteps, endpoint=False)
        d = posB - posA
        return [posA + l * d for l in lam]


    def integrate(self):
        '''
        Perform the integration along the path specified in the constructor
        :return: maximum energy error along path and range of the energy along path
        '''
        x = self.center.copy()
        x.calc = self.center.calc # todo: is this safe? MG: Yes

        if self.verbose:
            print('== Path Integration Test ==')
            print('  units: Angstrom, eV')
            print('  radius of circle: {}'.format(self.radius))
            print('  number of steps: {}'.format(self.nsteps))
            print('')
            print('  -- begin path integration --')
            print('     step |            energy | integrated energy |           error')

        for i, pos in enumerate(self.positions):
            if self.check_stress:
                reduced_positions = self.center.get_positions() @ np.linalg.inv(self.center.get_cell(True))
                x.set_positions(reduced_positions @ pos)
                x.set_cell(pos)
            else:
                x.set_positions(pos)
            if i == 0:
                self.e0 = x.get_potential_energy()
            self.__integration_step(x)

        energy_range = np.max(self.energies) - np.min(self.energies)
        self.energy_error = np.array(self.integrated_energies) - np.array(self.energies)
        max_error = np.max(np.abs(self.energy_error))

        if self.verbose:
            print('  -- end path integration --')
            print('')
            print('  maximum error: {}'.format(max_error))
            print('  energy range:  {}'.format(energy_range))
            print('  ratio:         {}'.format(max_error / energy_range))
            print('== End Path Integration Test ==')
        return max_error, energy_range


    def __get_energy_and_derivative(self, x: Atoms):
        e = x.get_potential_energy()
        if self.check_stress:
            return e, self.lattice_derivative(x.get_stress(voigt=False), x.get_cell(True))
        else:
            return e, -1. * x.get_forces()

    def __integration_step(self, x):
        i = len(self.energies)
        energy, energy_derivative = self.__get_energy_and_derivative(x)
        energy = energy - self.e0
        self.energies.append(energy)
        self.energy_derivatives.append(energy_derivative)
        if i == 0:
            self.integrated_energies.append(self.energies[0])
        else:
            f_mean = (self.energy_derivatives[i] + self.energy_derivatives[i-1]) / 2
            #f_mean = self.energy_derivatives[i-1]
            dx = self.positions[i] - self.positions[i-1]
            self.integrated_energies.append(self.integrated_energies[i-1] + np.sum(f_mean * dx))
        if self.verbose:
            print('    {:5d} {:19.7f} {:19.7f} {:17.11f}'.format(
                i,
                self.energies[i],
                self.integrated_energies[i],
                self.integrated_energies[i] - self.energies[i]))

    def __get_path_length(self):
        if self.shape == 'circle':
            return np.linspace(0, 2 * np.pi * self.radius, self.nsteps + 1)
        elif self.shape == 'line':
            return np.linspace(0, self.radius, self.nsteps)
        elif self.shape == 'pentagram':
            corners = self.__circle(5)
            l_line = np.linalg.norm(corners[2] - corners[0])
            nsteps_circ, nsteps_line = self.__get_pentagram_nsteps(self.nsteps)
            path = list(np.linspace(0, 2 * np.pi * self.radius, nsteps_circ, endpoint=False))
            for i in range(5):
                if i == 4:
                    path.extend(2 * np.pi * self.radius + l_line * i + np.linspace(0, l_line, nsteps_line + 1, endpoint=True))
                else:
                    path.extend(2 * np.pi * self.radius + l_line * i + np.linspace(0, l_line, nsteps_line, endpoint=False))
            #plt.plot(range(len(path)), path)
            #plt.show()
            return path

    def plot_pentagram_energy(self):
        '''
        If the pentagram option was choosen and the center is at a local minimum this creates a nice star shaped plot
        :return:
        '''
        import matplotlib.pyplot as plt
        nsteps_circ, nsteps_line = self.__get_pentagram_nsteps(self.nsteps)
        path = list(np.linspace(0, 2 * np.pi, nsteps_circ, endpoint=False))
        for i in range(5):
            if i == 4:
                path.extend((2 / 5 * i + 1) * 2 * np.pi + np.linspace(0, 4 / 5 * np.pi, nsteps_line + 1, endpoint=True))
            else:
                path.extend((2 / 5 * i + 1) * 2 * np.pi + np.linspace(0, 4 / 5 * np.pi, nsteps_line, endpoint=False))
        plt.polar(path, self.energies, label='energy')
        plt.polar(path, self.integrated_energies, label='integrated energy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_pentagram_error(self):
        import matplotlib.pyplot as plt
        nsteps_circ, nsteps_line = self.__get_pentagram_nsteps(self.nsteps)
        path = list(np.linspace(0, 2 * np.pi, nsteps_circ, endpoint=False))
        for i in range(5):
            if i == 4:
                path.extend((2 / 5 * i + 1) * 2 * np.pi + np.linspace(0, 4 / 5 * np.pi, nsteps_line + 1, endpoint=True))
            else:
                path.extend((2 / 5 * i + 1) * 2 * np.pi + np.linspace(0, 4 / 5 * np.pi, nsteps_line, endpoint=False))
        plt.polar(path, self.energy_error, label='energy error')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_energy_along_path(self):
        '''
        Plots the energy and the integrated energy along the path to visually check if the forces are correct
        :return:
        '''
        import matplotlib.pyplot as plt
        path = self.__get_path_length()
        plt.plot(path, self.energies, label='energy')
        plt.plot(path, self.integrated_energies, label='integrated energy')
        plt.xlabel(r'path length [$\rm \AA$]')
        plt.ylabel(r'energy [eV]')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_error_along_path(self):
        '''
        Plots the error along the path (integrated_energy - real_energy)
        :return:
        '''
        import matplotlib.pyplot as plt
        path = self.__get_path_length()
        plt.grid()
        plt.plot(path, self.energy_error, label='energy error')
        plt.xlabel(r'path length [$\rm \AA$]')
        plt.ylabel(r'energy [eV]')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('# iteration, energy, integrated energy, energy error, energy derivative norm (all units are Angstrom and eV)\n')
            for i in range(len(self.positions)):
                f.write(f'{i} {self.energies[i]} {self.integrated_energies[i]} {self.energy_error[i]} '
                         + f'{np.linalg.norm(self.energy_derivatives[i])}\n')

    def write_trajectory(self, filename):
        '''
        Writes the trajectory coordinates to a file
        :param filename:
        :return:
        '''
        # todo: lattice?
        def set_pos(pos):
            ats = self.center.copy()
            if self.check_stress:
                reduced_positions = self.center.get_scaled_positions()
                ats.set_cell(pos)
                ats.set_scaled_positions(reduced_positions)
                
            else:
                ats.set_positions(pos)
            return ats

        atoms_list = [set_pos(pos) for pos in self.positions]
        write(filename, atoms_list)




    @staticmethod
    def lattice_derivative(stress_tensor, cell):
        """
        Calculation of the lattice derivative from the stress tensor. This function cannot be used or has to be changed
        if the stress tensor is not included in the calculator used
        Input:
            stress_tensor: stress tensor from ase atoms object
                stress tensor from ase atoms object (atoms.get_stress(voigt=False,apply_constraint=False))
            cell: cell from ase atoms object (atoms.get_cell(complete=True))
        Return:
            deralat: np array
                numpy array containing the lattice derivatives
        """
        assert stress_tensor.shape == (3,3), 'Stress tensor is not a 3x3 array'
        assert cell.shape == (3,3), 'Cell is not a 3x3 array'

        inv_cell = np.linalg.inv(cell)
        prefact = np.linalg.det(cell)
        deralat = prefact * np.matmul(stress_tensor, inv_cell)
        return deralat
