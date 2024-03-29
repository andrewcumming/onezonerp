from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
	cmdclass = {'build_ext': build_ext},
	ext_modules = [Extension("net", ["net.pyx"]),
		Extension("kappa",["kappa.pyx"]), Extension("eos",["eos.pyx"])]
	)

