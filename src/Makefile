CFLAGS = -I../include/ -O3 -ffast-math -fsingle-precision-constant -msse2 -pthread

.PHONY: all clean

all: ../libpycann.so

clean:
	rm -f *.o ../libpycann.so

../libpycann.so: pycann.c
	$(CC) -shared -Wl,-soname,libpycann.so $(CFLAGS) -o $@ $^ -lc

pycann.s: pycann.c
	$(CC) -c -S $(CFLAGS) -o $@ $^