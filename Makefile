CXX        = g++
CXXFLAGS   = -Wall -Wextra -Wpedantic
OPTFLAGS   = -O0
PROFFLAGS  = -pg
INCLUDE    = -I ./include
PYTHON     = python3.8
LIBS       = -lglfw -lEGL -lGL


src        = $(wildcard src/*.cpp)
obj        = $(patsubst src/%.cpp,build/%.o,$(src))


targets += porting


all: $(targets)


porting: $(obj)
	$(CXX) -o $@ $^ $(LIBS)


output.bin: porting
	./$^ $@


regression: output.bin
	$(PYTHON) python/compare.py reference.bin $^


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


profile:
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(PROFFLAGS) $(INCLUDE) -o $@ $(src) $(LIBS)


# (INCOMPLETE) in the future this will definetely be automatically generated
build/%.o: src/%.cpp include/utils.hpp include/slm.hpp include/units.hpp
	$(CXX) -c $(CXXFLAGS) $(OPTFLAGS) $(INCLUDE) -o $@ $<


.PHONY clean:
clean:
	rm -f $(obj) porting profile output.bin
