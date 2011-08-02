NBC = nbc
NBC_FLAGS = -I=../../embedded/nxt/ -EF -Z2 -sm-
PYTHON31 = /usr/bin/python3.1

NETFILES = and.pcn and.rnn \
           or.pcn  or.rnn \
           xor.pcn xor.rnn



.PHONY: all clean install_nxt

all: $(NETFILES)

%.pcn: create.py
	$(PYTHON31) $^
%.rnn: create.py
	$(PYTHON31) $^

clean:
	rm -f $(NETFILES)
	rm -f test_logic.rxe

test_logic.rxe: test.nxc
	$(NBC) $(NBC_FLAGS) -O=$@ $^

install_nxt: test_logic.rxe $(NETFILES)
	nxt_upload -f or.rnn || true
	nxt_upload -f and.rnn || true
	nxt_upload -f xor.rnn || true
	nxt_upload -f test_logic.rxe || true


