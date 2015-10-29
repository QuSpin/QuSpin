from BitOps import *


class SpinOpError(Exception):
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message




def SpinOp(s,opstr,indx,pauli=False):
	if len(indx) != len(opstr):
		raise SpinOpError("Dimension mismatch of opstr and indx")

	ME=1.0
	r=s
	Nops=len(opstr)
	if pauli:
		for i in xrange(Nops-1,-1,-1): #string is written left to right, but operators act in reverse order.
			if opstr[i] == "c":
				continue
			elif opstr[i] == "z":
				a=testBit(r,indx[i])
				ME *= (-1)**(a+1)
			elif opstr[i] == "x":
				r=flipBit(r,indx[i])
			elif opstr[i] == "y":
				a=testBit(r,indx[i])
				r=flipBit(r,indx[i])
				ME *= (-1)**(a+1) * 1j
			elif opstr[i] == "+":
				a=testBit(r,indx[i])
				if a == 1: r=s; ME=0.0; break # + operator kills the state --> end loop
				else: r=flipBit(r,indx[i])
				ME *= 2.0
			elif opstr[i] == "-":
				a=testBit(r,indx[i])
				if a == 0: r=s; ME=0.0; break # - operator kills the state --> end loop
				else: r=flipBit(r,indx[i])
				ME *= 2.0
			else:
				raise SpinOpError("operator symbol "+opstr[i]+" not recognized")
	else:
		for i in xrange(Nops-1,-1,-1): #string is written left to right, but operators act in reverse order.
			if opstr[i] == "c":
				continue
			elif opstr[i] == "z":
				a=testBit(r,indx[i])
				ME*=(-1)**(a+1)/2.0
			elif opstr[i] == "x":
				r=flipBit(r,indx[i])
				ME*=1/2.0
			elif opstr[i] == "y":
				a=testBit(r,indx[i])
				r=flipBit(r,indx[i])
				ME*=(-1)**(a+1)*1j/2.0
			elif opstr[i] == "+":
				a=testBit(r,indx[i])
				if a == 1: r=s; ME=0.0; break # + operator kills the state --> end loop
				else: r=flipBit(r,indx[i])
			elif opstr[i] == "-":
				a=testBit(r,indx[i])
				if a == 0: r=s; ME=0.0; break # - operator kills the state --> end loop
				else: r=flipBit(r,indx[i])
			else:
				raise SpinOpError("operator symbol "+opstr[i]+" not recognized")

#	print opstr, indx, ME, s, r
	if ME.imag == 0.0:
		ME=ME.real

	return ME,r

