CXX      = g++
CXXFLAGS = -Wall -Wextra -Wpedantic
INCLUDE  = -I ./include
PYTHON   = python3.8


src = src/main.cpp src/serial.cpp
obj = $(patsubst src/%.cpp,build/%.o,$(src))


all: porting


example:
	$(PYTHON) python/example.py


porting: $(obj)
	$(CXX) -o $@ $^


output.bin: porting
	./$^ $@


regression: output.bin
	diff reference.bin output.bin


report: output.bin
	$(PYTHON) python/report.py $^


build/%.o: src/%.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDE) -o $@ $^


.PHONY clean:
clean:
	rm -f $(obj) porting output.bin
