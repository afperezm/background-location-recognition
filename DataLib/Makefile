# Makefile for DataLib

OPTFLAGS=-O2
OTHERFLAGS=-g -Wall -Wno-unused-result -c -fmessage-length=0 -std=c++0x

INCLUDE_PATH=-I../../VocabTree2

OBJS=reader.o

CPPFLAGS=$(INCLUDE_PATH) $(OTHERFLAGS) $(OPTFLAGS)

LIB=libreader.a

all: $(LIB)

$(LIB): $(OBJS)
	ar r $(LIB) $(OBJS)
	cp $(LIB) ..

clean:
	rm -f *.o *~ $(LIB)
