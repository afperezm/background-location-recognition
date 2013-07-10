# Makefile for Common

OPTFLAGS=-O2
OTHERFLAGS=-g -Wall -c -fmessage-length=0 -std=c++0x

INCLUDE_PATH=-I../../VocabTree2
INCLUDE_PATH+= `pkg-config opencv --cflags`

OBJS=FileUtils.o StringUtils.o

CPPFLAGS=$(INCLUDE_PATH) $(OTHERFLAGS) $(OPTFLAGS)

LIB=libcommon.a

all: $(LIB)

$(LIB): $(OBJS)
	ar r $(LIB) $(OBJS)
	cp $(LIB) ..

clean:
	rm -f *.o *~ $(LIB)
