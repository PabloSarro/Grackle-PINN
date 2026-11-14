#!/bin/bash
# Builds the Grackle driver on Izar against the Grackle install of the Lessandre project.
# The spack paths below are those printed by `ldd` for the libraries the install links to (HDF5, gfortran).
G=/home/sarro/SemProj/Lessandre/grackle-install
H=/ssoft/spack/syrah/v2/opt/spack/linux-rhel9-skylake_avx512/gcc-11.3.0/hdf5-1.12.2-eym7bvsim6gx7dzgghwmh6p3d3qqimpv
F=/ssoft/spack/syrah/v2/opt/spack/linux-rhel9-x86_64_v2/gcc-11.2.1/gcc-11.3.0-xt3pxujartqg46r6lercdmcya5bcxth4/lib64
cd "$(dirname "$0")"
gcc -O2 -o cooling-full-integration-100yr cooling-full-integration-100yr.c -I$G/include -L$G/lib64 -lgrackle \
    -L$H/lib -lhdf5 -L$F -lgfortran -lm -Wl,-rpath,$H/lib:$F
