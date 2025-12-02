# =============================
# User-configurable NuFast path
# =============================

NUFAST_DIR ?= ../NuFast-Earth

# =============================
# Compiler options
# =============================

CXX        := g++
CXXFLAGS   := -O3 -std=c++17 -fPIC -I$(NUFAST_DIR)/include

# =============================
# Output shared library
# =============================

TARGET_LIB := src/oscillations/libnufast_earth.so

# =============================
# Wrapper file (your code)
# =============================

WRAPPER_DIR := src/oscillations
WRAPPER_SRC := $(WRAPPER_DIR)/nuFast_wrapper.cpp
WRAPPER_OBJ := $(WRAPPER_DIR)/nuFast_wrapper.o

# =============================
# Default target
# =============================

all: $(TARGET_LIB)

# =============================
# Rule: build libnufast_earth.so
# =============================

$(TARGET_LIB): $(WRAPPER_OBJ)
	@echo "==> Building NuFast-Earth with -fPIC"
	$(MAKE) -C $(NUFAST_DIR) CFlags="-c -O3 -MMD -std=c++17 -fPIC"

	@echo "==> Linking $(TARGET_LIB)"
	$(CXX) -shared -o $@ \
	    $(WRAPPER_OBJ) \
	    $(NUFAST_DIR)/obj/*.o \
	    -lgsl -lgslcblas

# =============================
# Compile your wrapper
# =============================

$(WRAPPER_OBJ): $(WRAPPER_SRC)
	@echo "==> Compiling wrapper"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# =============================
# Cleanup
# =============================

clean:
	rm -f $(WRAPPER_OBJ) $(TARGET_LIB)

.PHONY: all clean
