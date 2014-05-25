PREFIX=/home/jure/build/torch7
CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lcublas -lluaT -lTHC -lTH

OBJ = adcensus.o

%.o : %.cu
	nvcc --compiler-options -Wall -arch sm_35 --compiler-options '-fPIC' -c $(CFLAGS) $<

libadcensus.so: ${OBJ}
	nvcc -arch sm_35 -o libadcensus.so --shared ${OBJ} $(LDFLAGS)

clean:
	rm -f *.o *.so
