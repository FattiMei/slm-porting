CXX      = g++
CXXFLAGS = -Wall -Wextra -Wpedantic
OPTFLAGS = -O2
INCLUDE  = -I ./include
PYTHON   = python3.8


src = src/main.cpp src/serial.cpp src/utils.cpp
obj = $(patsubst src/%.cpp,build/%.o,$(src))


all: porting


example:
	$(PYTHON) python/example.py


porting: $(obj)
	$(CXX) -o $@ $^


output.bin: porting
	./$^ $@


regression: output.bin
	$(PYTHON) python/compare.py reference.bin $^


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^



build/%.o: src/%.cpp
	$(CXX) -c $(CXXFLAGS) $(OPTFLAGS) $(INCLUDE) -o $@ $^


.PHONY clean:
clean:
	rm -f $(obj) porting output.bin
