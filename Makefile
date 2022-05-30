CXX = clang++
CFLAGS = -shared -std=c++17 -fPIC -m64
PWD = $(shell pwd)

INC = -I/usr/include/python3.8 -I/usr/include/pybind11 -lpython3.8
NUMPY = -I`python3 -c 'import numpy;print(numpy.get_include())'`

_matrix:
	$(CXX) $(CFLAGS) $(NUMPY) -O3 `python3 -m pybind11 --includes` yablas.cpp -o yablas`python3-config --extension-suffix` $(INC) 

.PHONY: test clean

clean:
	rm -rf *.so __pycache__ .pytest_cache