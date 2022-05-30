CXX = clang++
CFLAGS = -shared -std=c++17 -fPIC -m64
PWD = $(shell pwd)

INC = -I/usr/include/python3.8 -I/usr/include/pybind11 -lpython3.8
NUMPY = -I`python3 -c 'import numpy;print(numpy.get_include())'`

_yablas:
	$(CXX) $(CFLAGS) $(NUMPY) -Ofast `python3 -m pybind11 --includes` yablas.cpp -o yablas`python3-config --extension-suffix` $(INC) 

.PHONY: test clean

test:
	python3 -m pytest

clean:
	rm -rf *.so __pycache__ .pytest_cache