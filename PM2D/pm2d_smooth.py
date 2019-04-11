# -*- coding:utf-8 -*-

import sys
import numpy as np
from   numpy import random, fft
import scipy.ndimage as ndimage

random.seed(1)

def subdiv_unitcell_gen(n):
	subdiv = np.array( \
		[[0.0, 0.0],
		 [0.0, 0.5],
		 [0.5, 0.0],
		 [0.5, 0.5]])

	if (n == 0):
		return np.array([0,0])
	else:
		P = subdiv[:,np.newaxis] + \
			0.5 * subdiv_unitcell_gen(n-1)[np.newaxis,:]
		return P.reshape((P.size/2, 2))

subdiv_unitcell = [subdiv_unitcell_gen(n) for n in range(8)]

class Interp2D:
    "Reasonably fast bilinear interpolation routine"
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __call__(self, x):
        X1 = np.floor(x).astype(int) % self.shape
        X2 = np.ceil(x).astype(int) % self.shape
        xm = x % 1.0
        xn = 1.0 - xm

        f1 = self.data[X1[:,0], X1[:,1]]
        f2 = self.data[X2[:,0], X1[:,1]]
        f3 = self.data[X1[:,0], X2[:,1]]
        f4 = self.data[X2[:,0], X2[:,1]]

        return  f1 * xn[:,0] * xn[:,1] + \
                f2 * xm[:,0] * xn[:,1] + \
                f3 * xn[:,0] * xm[:,1] + \
                f4 * xm[:,0] * xm[:,1]

def gradient_dual(F, i):
	return F - np.roll(F, 1, axis=i)

def gradient_2nd_order(F, i):
	return   1./12 * np.roll(F,  2, axis=i) - 2./3  * np.roll(F,  1, axis=i) \
		   + 2./3  * np.roll(F, -1, axis=i) - 1./12 * np.roll(F, -2, axis=i)

class Box:
	"""This object keeps all knowledge about box-sizes and resolutions."""
	def __init__(self, Nm, Nf, L, qpf = 3):
		self.Nm = Nm
		self.Nf = Nf
		self.L = L
		self.qpf = qpf

		self.mass_res = L / Nm
		self.force_res = L / Nf

		self.X = np.indices((Nm, Nm)).astype(float)
		self.MX = self.X * self.mass_res

		# subdivide mass elements
		self.XX = (self.X.transpose([1,2,0]).reshape([Nm**2,2])[:,np.newaxis,:] + \
				subdiv_unitcell[self.qpf]).reshape([Nm**2 * 2**(2*self.qpf), 2])

		self.FX = np.indices((Nf, Nf)).astype(float) * self.force_res

		# k-values, divide by resolution to get physical scales
		self.Km = self.make_K(Nm)
		self.km2 = (self.Km**2).sum(axis=0)
		self.km2[0, 0] = 1

		self.Kf = self.make_K(Nf)
		self.kf2 = (self.Kf**2).sum(axis=0)
		self.kf2[0, 0] = 1

	@staticmethod
	def make_K(N):
		idx = np.indices((N, N))
		return np.where(idx > N/2, idx - N, idx) * (2*np.pi / N)

	def generate_init_density(self, P, F = lambda x: 1):
		"""generate initial condition with power-spectrum P and 
		post-filtering F (either gaussian smoothing or transfer function)"""
		wn = random.normal(0.0, 1.0, (self.Nm, self.Nm))
		km = np.sqrt(self.km2)
		fwn = fft.fftn(wn) * np.sqrt(P(km)) * F(self.Km)
		return fft.ifftn(fwn).real

	def compute_density(self, particles):
		"""compute density of partices on the force mesh. the partices are
		simply deposited into integer bins, using the searchsorted routine"""
		idc = ((particles / self.force_res) % self.Nf).astype(int)
		idx = np.sort(idc[:,0] * self.Nf + idc[:,1])
		cumdens = np.searchsorted(np.r_[idx,self.Nf**2], np.arange(self.Nf**2))
		idens =	(np.r_[cumdens[1:],idx.size]-cumdens).reshape( \
				(self.Nf,self.Nf))
		return idens.astype(float) / (2**(2*self.qpf)) * (self.mass_res / self.force_res)**2

	def compute_init_potential(self, Dens):
		"""compute velocity potential from initial density"""
		fDens = fft.fftn(Dens)
		return fft.ifftn(fDens / self.km2 * self.mass_res**2).real

	def compute_init_displacement(self, Dens):
		"""compute Zeldovich displacement from initial density"""
		fDens = fft.fftn(Dens)
		fPot  = fDens / self.km2 * self.mass_res**2
		vx = fft.ifftn(fPot * -1j * np.sin(self.Km[0])).real / self.mass_res
		vy = fft.ifftn(fPot * -1j * np.sin(self.Km[1])).real / self.mass_res
		return np.array([vx,vy])


class Cosmology:
	"""This object stores all relevant information wrt the background
	cosmology, parametrized by OmegaM, OmegaL and H0."""
	def __init__(self, H0, OmegaM, OmegaL):
		self.H0 = H0
		self.OmegaM = OmegaM
		self.OmegaL = OmegaL
		self.OmegaK = 1 - OmegaM - OmegaL
		self.grav_cst = 3./2 * OmegaM * H0**2

	def adot(self, a):
		return self.H0 * a * np.sqrt(self.OmegaL \
				+ self.OmegaM * a**-3 \
				+ self.OmegaK * a**-2)


class Solver:
	"""Solves the Poisson-Vlasov equation for a gravitating pressureless 
	medium, in an expanding Universe. The equations are
		∇²φ = 3/2 Ωm H0² δ/a
		  ṗ = -∇φ
		  ẋ = p/a² ┌───────────────────────┐
		  ȧ = H0 a ⎷ΩΛ + Ωm a⁻³ + (1-Ω) a⁻²
	solving for a by chain rule ẋ = ȧ dx/da."""
	def __init__(self, B, cosmology, dens, a_init):
		self.B = B
		self.cosmology = cosmology

		u = self.B.compute_init_displacement(dens)
		adot = cosmology.adot(a_init)

		# p = a²ẋ
		self.p = u * adot * a_init**2
		# acc = -a∇φ = aṗ
		self.acc = np.zeros_like(u)
		# psi = x - q
		self.psi = u * a_init

	def deploy(self):
		"""Deploy particles to compute densities. We interpolate a refined
		lagrangian mesh and deposit those particles."""
		psi_x = Interp2D(self.psi[0])
		psi_y = Interp2D(self.psi[1])
		qx = self.B.XX[:,0] * self.B.mass_res + psi_x(self.B.XX) 
		qy = self.B.XX[:,1] * self.B.mass_res + psi_y(self.B.XX)
		self.dens = self.B.compute_density(np.c_[qx,qy]) - 1

	def gravitate(self):
		"""Compute the gravitational acceleration from the density."""
		potF = - fft.fftn(self.dens) / self.B.kf2 \
				* self.B.force_res**2 * self.cosmology.grav_cst
		# potF = np.where(self.B.kf2 >= np.pi**2, 0, potF)
		potF[0,0] = 0

		self.pot = fft.ifftn(potF).real
		#self.E_acc_x = - gradient_2nd_order(self.pot, 0) \
		self.E_acc_x = - gradient_dual(self.pot, 0) \
				/ self.B.force_res
		self.E_acc_y = - gradient_dual(self.pot, 1) \
				/ self.B.force_res

		acc_x = Interp2D(self.E_acc_x)
		acc_y = Interp2D(self.E_acc_y)

		x = ((self.B.MX + self.psi) % self.B.L).transpose( \
				[1,2,0]).reshape((self.B.Nm**2, 2)) / self.B.force_res

		self.acc[0,:,:] = acc_x(x - [0, 0.5]).reshape((self.B.Nm, self.B.Nm))
		self.acc[1,:,:] = acc_y(x - [0.5, 0]).reshape((self.B.Nm, self.B.Nm))

	def kick(self, a, da):
		"""the kick"""
		adot = self.cosmology.adot(a)
		self.p += da / (adot * a) * self.acc

	def drift(self, a, da):
		"""the drift"""
		adot = self.cosmology.adot(a)
		self.psi += da / (adot * a**2) * self.p

	def first_step(self, a, da):
		self.deploy()
		self.gravitate()
		return self.next_step(a, da)

	def next_step(self, a, da):
		"""leap-frog algorithm, kick-drift-kick"""
		self.kick(a, da/2)
		self.drift(a, da)
		self.deploy()
		self.gravitate()
		self.kick(a + da, da/2)
		return a + da

	def save_snapshot(self, a):
		f = file("snap-%05u.dat" % int(round(a*1000)), 'w')
		adot = self.cosmology.adot(a)
		f.write("# a = %f, adot = %f\n" % (a, adot))
		np.savetxt(f, np.c_[  
                    (self.B.X[0] * self.B.mass_res).flat, 	
                    (self.B.X[1] * self.B.mass_res).flat,
                    self.psi[0].flat,
                    self.psi[1].flat])		
                #np.savetxt(f, np.c_[(self.B.XX[:,0]*self.B.mass_res).flat, (self.B.XX[:,1]*self.B.mass_res).flat])
		f.write("\n\n")
		f.close()
		
	def save_snapshot_bin(self, npart, box, OmegaM,OmegaL,H0, a):
                f = open("snap-%05u.xy" % int(round(a*1000)), "w")
		
                np.array([npart], dtype='int32').tofile(f)
                np.array([box,OmegaM,OmegaL,H0,a], dtype='float32').tofile(f)
                
                x = np.array((self.B.X[0] * self.B.mass_res).flat) + np.array(self.psi[0].flat)
                y = np.array((self.B.X[1] * self.B.mass_res).flat) + np.array(self.psi[1].flat)
                
                xy = np.array(zip(x,y))
                xy.astype(dtype='float32').tofile(f)
                f.close()

                
L   = 128.0  #--- box size
Nm  = 256    #--- particle grid size
Nf  = 512
Np  = Nm*Nm  #--- Number of particles

sigma_Mpc = 1.0
sigma_pix = sigma_Mpc*Nm/L

Sw =  0.05
n  = -0.98

H0     = 68.0
OmegaM = 0.32
OmegaL = 0.68

a_init = 0.04
da     = 0.01

def make_P(B, n):
	return lambda k: np.where(k > 0, (k/B.mass_res)**n, 0)

def scale_filter(t, gamma):
	return lambda K: np.exp(-(2 - gamma) * t + \
		(1 - gamma) * (np.cos(K[0]) + np.cos(K[1])) * \
	 	 t + gamma * np.cos(K[0]) * np.cos(K[1]) * t)

def make_F(B, S):
	return lambda k: np.exp(-0.5 * (S/B.mass_res)**2 * k**2)

def print_array(A):
	for l in A:
		print(" ".join(map(str, l)))
	print("\n\n")

if __name__ == "__main__":
        B = Box(Nm, Nf, L, 4)
        
        dens = B.generate_init_density(make_P(B, n), scale_filter(Sw / B.mass_res, 4.0/9.0))
        dens /= dens.std()
        dens *= 10
        
        #--- Smooth initial conditions
        dens = ndimage.gaussian_filter(dens, sigma=(sigma_pix), order=0)
        
        C = Cosmology(H0, OmegaM, OmegaL)
        
        S = Solver(B, C, dens, a_init)
        a = S.first_step(a_init, da)
        #S.save_snapshot(a)
        S.save_snapshot_bin(Np, L, OmegaM,OmegaL,H0, a)

        for i in range(100):
                sys.stderr.write('.')
                sys.stderr.flush()

                a = S.next_step(a, da)
                #S.save_snapshot(a)
                S.save_snapshot_bin(Np, L, OmegaM,OmegaL,H0, a)

# vim:ts=4:sw=4:tw=80
