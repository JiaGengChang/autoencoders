IDIR=include
SRCDIR=src
CC=gcc
CFLAGS=-I$(IDIR) 

LIBS=-lm

ODIR=obj
_OBJ1 = testperceptron.o perceptron.o
_OBJ2 = testmlp.o mlp.o
OBJ1 = $(patsubst %,$(ODIR)/%,$(_OBJ1))
OBJ2 = $(patsubst %,$(ODIR)/%,$(_OBJ2))

_DEPS1 = perceptron.h
_DEPS2 = mlp.h
DEPS1 = $(patsubst %,$(IDIR)/%,$(_DEPS1))
DEPS2 = $(patsubst %,$(IDIR)/%,$(_DEPS2))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

1.out: $(OBJ1)
	$(CC) -o $@ $^ $(CLAGS) $(LIBS)

2.out: $(OBJ2)
	$(CC) -o $@ $^ $(CLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~
