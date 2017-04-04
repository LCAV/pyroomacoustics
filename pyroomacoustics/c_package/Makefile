CC = gcc
CFLAGS = -Wall -O3
#CFLAGS = -Wall -g
INCLUDES =
LFLAGS =
LIBS = -lm

LIB_MAIN = libroom.so
LIB_HDR = room.h
LIB_SRC = wall.c linalg.c room.c is_list.c shoebox.c
LIB_OBJS = $(LIB_SRC:.c=.o)

TEST_MAIN = tests
TEST_SRC = tests.c
TEST_OBJS = $(TEST_SRC:.c=.o)

all: $(LIB_MAIN) $(TEST_MAIN)

$(LIB_MAIN): $(LIB_OBJS) 
			$(CC) -shared $(CFLAGS) $(INCLUDES) -o $(LIB_MAIN) $(LIB_OBJS) $(LFLAGS) $(LIBS)

$(TEST_MAIN): $(LIB_MAIN) $(TEST_OBJS) 
			$(CC) -L. $(CFLAGS) $(INCLUDES) -o $(TEST_MAIN) $(TEST_OBJS) $(LFLAGS) $(LIBS) -lroom

$(TEST_HASH_MAIN): $(LIB_MAIN) $(TEST_HASH_OBJS) 
			$(CC) -L. $(CFLAGS) $(INCLUDES) -o $(TEST_HASH_MAIN) $(TEST_HASH_OBJS) $(LFLAGS) $(LIBS) -lroom

share_lib: wall.o
	gcc -shared -o room.so 

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.c.o:
			$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
			$(RM) *.o *~ $(LIB_MAIN) $(TEST_MAIN)
