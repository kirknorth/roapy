# Remove and build _texture.so
rm -vf _texture.so
f2py -m _texture -h _texture.pyf _texture.f90 --overwrite-signature
f2py --fcompiler=gfortran -c _texture.pyf _texture.f90 
