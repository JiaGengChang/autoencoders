IDIR=include
SRCDIR=src
CC=gcc
CFLAGS=-I$(IDIR) 

LIBS=-lm

ODIR=obj
_OBJ1 = testperceptron.o perceptron.o
_OBJ2 = testmlp.o utils.o
_OBJ3 = testmlp16.o utils.o
OBJ1 = $(patsubst %,$(ODIR)/%,$(_OBJ1))
OBJ2 = $(patsubst %,$(ODIR)/%,$(_OBJ2))
OBJ3 = $(patsubst %,$(ODIR)/%,$(_OBJ3))

_DEPS1 = perceptron.h 
_DEPS2 = utils.h
_DEPS3 = utils.h
DEPS1 = $(patsubst %,$(IDIR)/%,$(_DEPS1))
DEPS2 = $(patsubst %,$(IDIR)/%,$(_DEPS2))
DEPS3 = $(patsubst %,$(IDIR)/%,$(_DEPS3))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

perceptron: $(OBJ1)
	$(CC) -o $@ $^ $(CLAGS) $(LIBS)

mlp: $(OBJ2)
	$(CC) -o $@ $^ $(CLAGS) $(LIBS)

mlp16: $(OBJ3)
	$(CC) -g -o $@ $^ $(CLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~
