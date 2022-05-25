#
# 'make depend' uses makedepend to automatically generate dependencies
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files

CUDA_OBJ = obj/cuda.o

# Input Names
CUDA_FILES = src/planet_graphics.cu src/noise.cu

# ------------------------------------------------------------------------------

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr -rdc=true
NVCC_INCLUDE =
NVCC_LIBS =
NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_52,code=sm_52 \
                -gencode arch=compute_60,code=sm_60 \
                -gencode arch=compute_61,code=sm_61 \
                -gencode arch=compute_61,code=compute_61

# CUDA Object Files
# CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(CUDA_FILES)))
CUDA_OBJ_FILES = obj/planet_graphics.cu.o obj/noise.cu.o

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# define the C compiler to use
CC = g++

# define any compile-time flags
CFLAGS = -Wall -g -O3 -pthread -D_REENTRANT -fpermissive

# define any directories containing header files other than /usr/include
INCLUDES = -I./include -I$(CUDA_INC_PATH)

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
# LFLAGS = -L./lib

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname
#   option, something like (this will link in libmylib.so and libm.so:
LIBS = -lm -L$(CUDA_LIB_PATH) -lcudart -lcufft -lsndfile

# define the C source files
SRCS = src/%.c lib/%.c

# define the C object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
# OBJS = $(SRCS:.c=.o)
LIBOBJS = obj/libattopng.o obj/noise.o obj/gifenc.o
OBJS = $(LIBOBJS) obj/mars.o obj/earth.o obj/planet_graphics.o obj/palette.o

#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: clean #depend

all: terminal imgtest
	@echo "Simple compiler named mycc has been compiled"

terminal: bin/animated_terminal
imgtest: bin/render_planet

bin/animated_terminal: $(OBJS) obj/animated_terminal.o
	$(CC) $(CFLAGS) $(INCLUDES) -o bin/animated_terminal $(OBJS) obj/animated_terminal.o $(LFLAGS) $(LIBS)

bin/render_planet: $(OBJS) $(CUDA_OBJ_FILES) $(CUDA_OBJ) obj/main.o
	$(CC) $(CFLAGS) $(INCLUDES) -o bin/render_planet $(OBJS) $(CUDA_OBJ_FILES) $(CUDA_OBJ) obj/main.o $(LFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
# %.o : %.c
# $(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@
obj/%.o: src/%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

obj/libattopng.o: lib/libattopng.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@
obj/noise.o: lib/noise.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

# MUST BE COMPILED AS C CODE (NOT C++)
obj/gifenc.o: lib/gifenc.c
	gcc -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

# Compile CUDA Source Files
obj/%.cu.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(INCLUDES) $(NVCC_INCLUDE) $<

cuda: $(CUDA_OBJ_FILES) $(CUDA_OBJ)

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^

clean:
	$(RM) obj/*.o *~ $(MAIN)
