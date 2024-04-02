CXX        = g++
CXXFLAGS   = -Wall -Wextra -Wpedantic
OPTFLAGS   = -O0
PROFFLAGS  = -pg
INCLUDE    = -I ./include
PYTHON     = python3.8
LIBS       = -lglfw -lEGL -lGL


all        = $(wildcard src/*.cpp)
obj        = $(patsubst src/%.cpp,build/%.o,$(all))


core_src   = src/serial.cpp src/units.cpp src/utils.cpp
core_obj   = $(patsubst src/%.cpp,build/%.o,$(core_src))


render_src = src/render.cpp src/shader.cpp src/texture.cpp src/window.cpp
render_obj = $(patsubst src/%.cpp,build/%.o,$(render_src))


targets += porting interactive


all: $(targets)


porting: $(core_obj) build/main.o build/texture.o
	$(CXX) -o $@ $^ $(LIBS)


interactive: $(core_obj) $(render_obj) build/interactive.o
	$(CXX) -o $@ $^ $(LIBS)


run: interactive
	./$^


output.bin: porting
	./$^ $@


regression: output.bin
	$(PYTHON) python/compare.py reference.bin $^


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


profile:
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(PROFFLAGS) $(INCLUDE) -o $@ $(src)


# (INCOMPLETE) in the future this will definetely be automatically generated
build/%.o: src/%.cpp include/utils.hpp include/slm.hpp include/units.hpp
	$(CXX) -c $(CXXFLAGS) $(OPTFLAGS) $(INCLUDE) -o $@ $<


.PHONY clean:
clean:
	rm -f $(obj) porting profile output.bin
