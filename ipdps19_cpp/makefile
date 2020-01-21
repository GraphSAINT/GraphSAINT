CC=icc
IDIR=./include
ODIR=./obj
LIBS=-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 
CFLAGS=-I${IDIR} -I${MKLROOT}/include -fopenmp -pthread -Wall -O3 --std=c++11

_DEPS=global.h optm.h		# global dependencies
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ=init.o layer.o operation.o util.o sample.o optm.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cpp $(IDIR)/%.h $(DEPS)
	@mkdir -p $(@D)
	$(CC) -c -o $@ $< $(CFLAGS)

train: train.cpp $(OBJ)
	$(CC) train.cpp -o $@ $(OBJ) $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o

