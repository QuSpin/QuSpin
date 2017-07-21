import numpy as _np


class site_info(object):
	def __init__(self,N):
		self._N = N
		self._sites = _np.arange(N)

	@property
	def N(self):
		return self._N

	@property
	def sites(self):
		return self._sites

class site_info_2d(site_info):
	def __init__(self):
		pass

	@property
	def coor_iter(self):
		return enumerate(zip(self._X,self._Y))

class site_info_square(site_info_2d):
	def __init__(self,Lx,Ly):
		self._Lx = Lx
		self._Ly = Ly
		site_info.__init__(self,Lx*Ly)

		self._X = self.sites%Lx
		self._Y = self.sites/Lx

	@property
	def X(self):
		return self._X

	@property
	def Y(self):
		return self._Y


class square_lattice_trans(object):
	def __init__(self,Lx,Ly):
		
		self._site_info = site_info_square(Lx,Ly)
		sites = self.site_info.sites
		X = self.site_info.X
		Y = self.site_info.Y

		self._Z   = -(sites+1)
		self._Z_A = _np.array([(-(i+1) if (x+y)%2==0 else i) for i,(x,y) in enumerate(zip(X,Y))])
		self._Z_B = _np.array([(-(i+1) if (x+y)%2==1 else i) for i,(x,y) in enumerate(zip(X,Y))])
		self._T_x = (X+1)%Lx + Y*Lx
		self._T_y = X + ((Y+1)%Ly)*Lx
		self._P_x = (Lx-X-1) + Y*Lx
		self._P_y = X + (Ly-Y-1)*Lx
		if Lx==Ly:
			self._P_d = Y + Lx*X
			self._P_e = (Ly-Y-1) + Lx*(Lx-X-1)
		else:
			self._P_d = None
			self._P_e = None


	@property
	def site_info(self):
		return self._site_info

	@property
	def Z(self):
		return self._Z

	@property
	def Z_A(self):
		return self._Z_A

	@property
	def Z_B(self):
		return self._Z_B

	@property
	def T_x(self):
		return self._T_x

	@property
	def T_y(self):
		return self._T_y

	@property
	def P_x(self):
		return self._P_x

	@property
	def P_y(self):
		return self._P_y

	@property
	def P_e(self):
		if self._P_e is None:
			raise Exception("P_e symmetry exsits for square lattice")

		return self._P_e

	@property
	def P_d(self):
		if self._P_d is None:
			raise Exception("P_d symmetry exsits for square lattice")

		return self._P_d
