from BitOps import int2bin


class SpinOpError(Exception):
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message




def SpinOp(s,opstr,indx):
	if len(indx) != len(opstr):
		raise SpinOpError("Dimension mismatch of opstr and indx")

	ME=1.0; r=s
	Nops=len(opstr)
	for i in xrange(Nops-1,-1,-1): #string is written left to right, but operators act in reverse order.
		t=r;a=1
		if opstr[i] == "c":
			continue
		elif opstr[i] == "z":
			a = ((r>>indx[i])&1)
			ME *= (-1)**(a+1)/2.0
		elif opstr[i] == "x":
			r = r^(1<<indx[i])
			ME *= 1/2.0
		elif opstr[i] == "y":
			a = ((r>>indx[i])&1)
			r = r^(1<<indx[i])
			ME *= (-1)**(a)/2.0j
		elif opstr[i] == "+":
			a = ((r>>indx[i])&1)
			if a == 1: r=s; ME=0.0; break # + operator kills the state --> end loop
			else: r = r^(1<<indx[i])
		elif opstr[i] == "-":
			a = ((r>>indx[i])&1)
			if a == 0: r=s; ME=0.0; break # - operator kills the state --> end loop
			else: r = r^(1<<indx[i])
		else:
			raise SpinOpError("operator symbol "+opstr[i]+" not recognized") 


	if ME.imag == 0.0:
		ME=ME.real
	return ME,r

