from numpy import array, arange

class BitOpsError(Exception):
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message





def int2bin(n,L):
	""" Convert an integer n to a binary vector 
	padded with zeros up to the appropriate length L """

	return (((fliplr(n,L) & (1 << arange(L)))) > 0).astype(int)

def bin2int(v):
	""" Convert a binary vector v to an integer """

	return np.sum([v[i]*2**i for i in xrange(len(v))])




def shift(int_type,shift,period):
# this functino is used to shift the bits of an integer by 'shift' bits.
# it is used when constructing the momentum states
	Imax=2**period-1
	if int_type==0 or int_type==Imax:
		return int_type
	else:
		if shift < 0:
			shift=-shift
			return ((int_type & (2**period-1)) >> shift%period) | (int_type << (period-(shift%period)) & (2**period-1))
		else:
			return (int_type << shift%period) & (2**period-1) | ((int_type & (2**period-1)) >> (period-(shift%period)))


def fliplrup(int_type,length,axis):
	M=[[x+length*y for x in xrange(length)] for y in xrange(length)]
	if axis == 1:
		for m in M:
			m.reverse()
	elif axis == 2:
		M=array(M).T.tolist()
		for m in M:
			m.reverse()
		M=array(M).T.tolist()

	new_state=0
	for y in xrange(length):
		for x in xrange(length):
			i=x+length*y
			j=M[y][x]
			if int_type>>i&1:
				new_state+=2**j

	return new_state
			


def fliplr(int_type,length):
# this function flips the bits of an integer around the centre, e.g. 1010 -> 0101
# (generator of) parity symmetry
    return sum(1<<(length-1-i) for i in xrange(length) if int_type>>i&1)

def flip_all(int_type,length):
# this function flips all bits
# (generator of) inversion symmetry
    lower = 0;
    upper = length;
    return int_type^((1<<upper)-1)&~((1<<lower)-1)




def testBit(int_type, i):
# returns whether the bit at 'i' is 0 or 1
	return (int_type>>i)&1

def sumBits(int_type,max_bit):
# sums all the bits of 'int_type' up to bit numeber 'max_bit'
	return sum(map(lambda x:(int_type>>x)&1, [ i for i in xrange(max_bit)]))

def sumStagBits(int_type,max_bit,Stag):
# does the same as sum bits, but one can put a staggered mask over the sum for things like staggaret magnetization
	return sum(map(lambda x:((int_type>>x)&1)*Stag[x], [ i for i in xrange(max_bit)]))

def exchangeBits(int_type,i,j):
# takes the bits i and j of 'int_type' and swaps their values, if the bits are both 0 or 1, it returns the same integer, else it returns the
# new integer after the flip.
	ibit=(int_type>>i)&1
	jbit=(int_type>>j)&1
	if ibit==jbit:
		return int_type
	else:
		return int_type^((1<<i)+(1<<j)) 


def flipBit(int_type,i):
# flips a single bit from 0->1 or vice versa at bit 'i'
	return int_type^(1<<i)
