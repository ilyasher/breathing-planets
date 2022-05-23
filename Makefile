#
# 'make depend' uses makedepend to automatically generate dependencies
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# define the C compiler to use
CC = gcc

# define any compile-time flags
CFLAGS = -Wall -g

# define any directories containing header files other than /usr/include
#
INCLUDES = -I./include

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
# LFLAGS = -L./lib

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname
#   option, something like (this will link in libmylib.so and libm.so:
LIBS = -lm

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

bin/render_planet: $(OBJS) obj/main.o
	$(CC) $(CFLAGS) $(INCLUDES) -o bin/render_planet $(OBJS) obj/main.o $(LFLAGS) $(LIBS)

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
obj/gifenc.o: lib/gifenc.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

clean:
	$(RM) obj/*.o *~ $(MAIN)
