
import numpy as _np
import warnings




class GeneralBasisWarning(Warning):
	pass

def process_map(map,q):
	map = _np.asarray(map,dtype=_np.int32)
	i_map = map.copy()
	i_map[map<0] = -(i_map[map<0] + 1) # site mapping
	s_map = map < 0 # sites with spin-inversion

	sites = _np.arange(len(map),dtype=_np.int32)
	order = sites.copy()

	if _np.any(_np.sort(i_map)-order):
		raise ValueError("map must be a one-to-one site mapping.")

	per = 0
	group = [tuple(order)]
	while(True):
		sites[s_map] = -(sites[s_map]+1)
		sites = sites[i_map]
		per += 1
		group.append(tuple(sites))
		if _np.array_equal(order,sites):
			break

	if per == 1:
		warnings.warn("identity mapping found in set of transformations.",GeneralBasisWarning,stacklevel=5)

	return map,per,q,set(group)

def check_symmetry_maps(item1,item2):
	grp1 = item1[1][-1]
	map1 = item1[1][0]
	block1 = item1[0]

	i_map1 = map1.copy()
	i_map1[map1<0] = -(i_map1[map1<0] + 1) # site mapping
	s_map1 = map1 < 0 # sites with spin-inversion		

	grp2 = item2[1][-1]
	map2 = item2[1][0]
	block2 = item2[0]

	i_map2 = map2.copy()
	i_map2[map2<0] = -(i_map2[map2<0] + 1) # site mapping
	s_map2 = map2 < 0 # sites with spin-inversion

	if grp1 == grp2:
		warnings.warn("mappings for block {} and block {} produce the same symmetry.".format(block1,block2),GeneralBasisWarning,stacklevel=5)

	sites1 = _np.arange(len(map1))
	sites2 = _np.arange(len(map2))

	sites1[s_map1] = -(sites1[s_map1]+1)
	sites1 = sites1[i_map1]
	sites1[s_map2] = -(sites1[s_map2]+1)
	sites1 = sites1[i_map2]

	sites2[s_map2] = -(sites2[s_map2]+1)
	sites2 = sites2[i_map2]
	sites2[s_map1] = -(sites2[s_map1]+1)
	sites2 = sites2[i_map1]

	if not _np.array_equal(sites1,sites2):
		warnings.warn("using non-commuting symmetries can lead to unwanted behaviour of general basis, make sure that quantum numbers are invariant under non-commuting symmetries!",GeneralBasisWarning,stacklevel=5)



