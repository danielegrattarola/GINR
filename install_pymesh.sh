set -e
set -u

cd $1
git clone https://github.com/PyMesh/PyMesh.git
cd PyMesh
git submodule update --init
cd third_party && ./build.py all && cd ../
./setup.py build
./setup.py install
