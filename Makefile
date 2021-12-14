IDIR=include
SRCDIR=src
CC=gcc
CFLAGS=-I$(IDIR) 

LIBS=-lm

ODIR=obj
_OBJ = testperceptron.o perceptron.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_DEPS = perceptron.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

a.out: $(OBJ)
	$(CC) -o $@ $^ $(CLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~
