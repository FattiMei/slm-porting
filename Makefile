CXX      = g++
CXXFLAGS = -Wall -Wextra -Wpedantic
PYTHON   = python3.8


src = src/main.cpp
obj = $(patsubst src/%.cpp,build/%.o,$(src))


all: porting


regression:
	$(PYTHON) python/regression.py


example:
	$(PYTHON) python/example.py


porting: $(obj)
	$(CXX) -o $@ $^


build/%.o: src/%.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $^
