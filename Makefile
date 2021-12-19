IDIR=include
SRCDIR=src
BIN=bin
CC=gcc
CFLAGS=-I$(IDIR) 

LIBS=-lm

ODIR=obj
_OBJ1 = testmlp.o utils.o
_OBJ2 = testmlp16.o utils.o
OBJ1 = $(patsubst %,$(ODIR)/%,$(_OBJ1))
OBJ2 = $(patsubst %,$(ODIR)/%,$(_OBJ2))

_DEPS = utils.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mlp: $(OBJ1)
	$(CC) -o $(BIN)/$@ $^ $(CLAGS) $(LIBS)

mlp16: $(OBJ2)
	$(CC) -o $(BIN)/$@ $^ $(CLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~
