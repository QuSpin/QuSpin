#

# -*- coding: utf-8 -*-
"""quspin_colab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16g_XBaabKWNomiXahzVk0K0ynsjyw790
"""

# Commented out IPython magic to ensure Python compatibility.
# # This codebox installs miniconda for python=3.10 and then quspin=0.3.7 with omp in colab.
#
# # If you would like to avoid installing quspin every new runtime, you can copy
# # the following code to a directory in your google drive.
#
# # !cp -r /usr/local/lib/python3.10/site-packages/quspin <your-directory>
# # !cp /usr/local/lib/python3.10/site-packages/gmpy2.cpython-37m-x86_64-linux-gnu.so <your-directory>
#
# # You will then need to add <your-directory> to sys.path just like above.
#
# %%bash
#
# # install conda for python 3.10
# MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.12-Linux-x86_64.sh
# MINICONDA_PREFIX=/usr/local
# wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
# chmod +x $MINICONDA_INSTALLER_SCRIPT
# ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
#
# # update conda
# conda install --channel defaults conda python=3.10 --yes
# conda update --channel defaults --all --yes
#
# # install quspin
# conda install -c weinbe58 quspin=0.3.7 omp --yes
#
# # check quspin installation
# import sys
# sys.path.append("/usr/local/lib/python3.10/site-packages")
# import quspin
# print('quspin {} installed successfully and ready to use'.format(quspin.__version__) )

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

basis = spin_basis_1d(L=2)
print(basis)

coupl = [
    [1.0, 0, 1],
]

static = [
    ["xx", coupl],
]

H = hamiltonian(static, [], basis=basis)
print(H.toarray())
