"""
Stochastic Advection Diffusion Equation solver
"""

import numpy as np
from scipy import linalg

class Stochastic_ADE:
    """
    Solves u(x)_t + a(x) u' = kappa(x) u''(x), where a(x) and kappa(x) are 2 pi periodic
    and randomly sampled using the Karhoenen-Loeve expansion.
    """

    def __init__(self, N, dt,
                 mean_a, sigma_a, l_a, T_a,
                 mean_kappa, sigma_kappa, l_kappa, T_kappa,
                 truncation_order_a, truncation_order_kappa,
                 **kwargs):
        """
        Create a Staochatic_ADE solver object.

        Parameters
        ----------
        N : int
            The number of grid points.
        dt : float
            The time step.
        mean_a : int
            The mean advection velocity
        sigma_a : float
            The scale of the auto-covariance function of a(x).
        l_a : float
            The correlation length of the auto-covariance function of a(x).
        T_a : float
            The period of the auto-covariance function of a(x).
        mean_kappa : int
            The mean diffusivity coefficient
        sigma_kappa : float
            The scale of the auto-covariance function of kappa(x).
        l_kappa : float
            The correlation length of the auto-covariance function of kappa(x).
        T_kappa : float
            The period of the auto-covariance function of kappa(x).
        truncation_order_a : int
            The number of terms to use in the Karhoenen-Loeve expansion of a(x).
        truncation_order_kappa : int
            The number of terms to use in the Karhoenen-Loeve expansion of kappa(x).
        kwargs : array, size (truncation_order,)
            The keywords z_a and z_kappa can be used to manually specify the values
            of the standard-normal input values of a(x) and kappa(x).

        Returns
        -------
        None.

        """

        self.N = N
        self.dt = dt

        self.mean_a = mean_a
        self.sigma_a = sigma_a
        self.l_a  = l_a
        self.T_a = T_a

        self.mean_kappa = mean_kappa
        self.sigma_kappa = sigma_kappa
        self.l_kappa  = l_kappa
        self.T_kappa = T_kappa

        self.truncation_order_a = truncation_order_a
        self.truncation_order_kappa = truncation_order_kappa

        # spatial grid
        self.x = np.linspace(0, 2 * np.pi, N)

        # random inputs for a(x), generated here or specified by user via kwargs
        z = kwargs.get('z_a', np.random.randn(truncation_order_a))

        # randomly sample a(x)
        self.eigenvalues_a, self.eigenvectors_a = self.karhoenen_loeve(sigma_a, l_a,
                                                                       T_a, truncation_order_a)
        self.sample_advection_velocity(z)

        # random inputs for kappa(x), generated here or specified by user via kwargs
        z = kwargs.get('z_kappa', np.random.randn(truncation_order_kappa))

        # randomly sample kappa(x)
        self.eigenvalues_kappa, self.eigenvectors_kappa = \
        self.karhoenen_loeve(sigma_kappa, l_kappa, T_kappa, truncation_order_kappa)
        self.sample_diffusivity(z)

        self.kx, self.k_squared = self.get_derivative_operators()

    def get_derivative_operators(self):
        """
        Get the spectral operators for the gradients in x direction.

        Returns
        -------
        kx : array (complex)
            Operator for derivative in x direction.
        k_squared : array (complex)
            Operator for 2nd derivative in x direction.

        """

        N = self.N
        # frequencies
        k = np.fft.fftfreq(N) * N

        kx = 1j * k
        k_squared = kx**2

        return kx, k_squared

    def initial_condition(self):
        """
        Get the initial condition.

        Returns
        -------
        float
            The IC of u(x).

        """

        u = 0.5 + 0.5* np.sin(self.x)
        return np.fft.fft(u)

    def rhs_hat(self, u_hat_n):
        dudx = np.fft.ifft(self.kx * u_hat_n)
        d2u_dx2 = np.fft.ifft(self.k_squared * u_hat_n)
        rhs_hat = np.fft.fft(-self.a * dudx + self.kappa * d2u_dx2) 
        return rhs_hat        
    

    def step_euler(self, u_hat_n):
        """
        Advance the system in time using forward Euler.

        Parameters
        ----------
        u_hat_n : complex array
            The Fourier coefficients of u(x) at time n.

        Returns
        -------
        complex array
            The Fourier coefficients of u(x) at time n + 1.

        """

        rhs_hat = self.rhs_hat(u_hat_n)
        return u_hat_n + rhs_hat * self.dt

    def step_rk4(self, u_hat_n):
        """
        Advance the system in time using 4-th order Runge Kutta.

        Parameters
        ----------
        u_hat_n : complex array
            The Fourier coefficients of u(x) at time n.

        Returns
        -------
        complex array
            The Fourier coefficients of u(x) at time n + 1.

        """
        # RK4 step 1
        k_hat_1 = self.rhs_hat(u_hat_n)
        k_hat_1 *= self.dt
        u_hat_2 = u_hat_n + k_hat_1 / 2
        # RK4 step 2
        k_hat_2 = self.rhs_hat(u_hat_2)
        k_hat_2 *= self.dt
        u_hat_3 = u_hat_n + k_hat_2 / 2
        # RK4 step 3
        k_hat_3 = self.rhs_hat(u_hat_3)
        k_hat_3 *= self.dt
        u_hat_4 = u_hat_n + k_hat_3
        # RK4 step 4
        k_hat_4 = self.rhs_hat(u_hat_4)
        k_hat_4 *= self.dt
        u_hat_np1 = u_hat_n  + 1 / 6 * (k_hat_1 + 2 * k_hat_2 + 2 * k_hat_3 + k_hat_4)

        return u_hat_np1

    def karhoenen_loeve(self, sigma, l, T, truncation_order):
        """
        Compute the eigenvectors and eigenvalues of the Karhoenen-Loeve expansion
        of the spatially varying periodic conduction coefficient. Stores values
        internally in self.eigenvalues and self.eigenvectors.

        Returns
        -------
        None.

        """
        N = self.N
        # create (auto)covariance matrix
        C = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                d = np.abs(self.x[i] - self.x[j])
                C[i, j] = sigma**2 *np.exp(- 2 * np.sin(np.pi * d / T)**2 / l**2)
        C_hat = np.fft.fft2(C)

        # the matrix B of the generalized eigenvalue problem
        B = np.zeros([N, N]) + 0.0j
        B[0, 0] = 1.0
        B[1:,1:] = np.flipud(np.eye(N-1))
        B *= 1 / (2 * np.pi)

        # solve the generalized eigenvalue problem Cv = lamda * Bv
        eigenvalues, eigenvectors = linalg.eig(C_hat, b=B)

        # Sort the eigensolutions in the descending order of eigenvalues
        order = eigenvalues.real.argsort()[::-1]
        eigenvalues = eigenvalues[order].real
        eigenvectors = eigenvectors[:, order]

        # Truncate the expansion
        eigenvalues = eigenvalues[:truncation_order]
        eigenvectors = eigenvectors[:, :truncation_order]

        return eigenvalues, eigenvectors

    def sample_advection_velocity(self, z):
        """
        Randomly sample the spatially varying advection velocity using the Karhoenen-Loeve
        expansion. Stores value internally in self.a.

        Parameters
        ----------
        z : array, size self.truncation_order
            Draws from indepedent standard normal random variables.

        Returns
        -------
        None.

        """

        a_hat = np.sum(np.sqrt(self.eigenvalues_a) * z * self.eigenvectors_a, axis = 1)
        self.a = self.mean_a + np.fft.ifft(a_hat).real

    def sample_diffusivity(self, z):
        """
        Randomly sample the spatially varying thermal diffusivity using the Karhoenen-Loeve
        expansion. Stores values internally in self.kappa.

        Parameters
        ----------
        z : array, size self.truncation_order
            Draws from indepedent standard normal random variables.

        Returns
        -------
        None.

        """

        kappa_hat = np.sum(np.sqrt(self.eigenvalues_kappa) * z * self.eigenvectors_kappa, axis = 1)
        self.kappa = self.mean_kappa + np.fft.ifft(kappa_hat).real
        print(self.eigenvalues_kappa)
        assert((self.kappa > 0).all()), "Diffusivity must be positive for all x"
