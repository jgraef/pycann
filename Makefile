PYTHON31 = /usr/bin/python3.1

.PHONY: all clean install

all:
	make -C src all
	make -C examples all

clean:
	make -C src clean
	make -C examples clean
	rm -f *.pyc

install:
	make -C src install
	$(PYTHON31) setup.py install
	cp include/pycann.h /usr/local/include
