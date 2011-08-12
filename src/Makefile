CFLAGS = -I../include/ -O3 -ffast-math -pthread -fPIC -fsingle-precision-constant -msse2

.PHONY: all clean install

all: ../libpycann.so

clean:
	rm -f ../libpycann.so

install: ../libpycann.so
	cp $< /usr/local/lib

../libpycann.so: pycann.c
	$(CC) -shared -Wl,-soname,libpycann.so $(CFLAGS) -o $@ $^ -lc

pycann.s: pycann.c
	$(CC) -c -S $(CFLAGS) -o $@ $^
