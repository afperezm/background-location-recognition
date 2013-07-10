# Makefile for ListBuild

OPTFLAGS=-O2
OTHERFLAGS=-g -Wall -fmessage-length=0 -std=c++0x

INCLUDE_PATH=-I../ -I../../VocabTree2
INCLUDE_PATH+= `pkg-config opencv --cflags`

LIB_PATH=-L../../VocabTree2 -L/usr/include -L../

OBJS=ListBuild.o lists_builder.o

LIBS=-lcommon -lreader -lvocab `pkg-config opencv --libs`

CPPFLAGS=$(INCLUDE_PATH) $(LIB_PATH) $(OTHERFLAGS) $(OPTFLAGS)

BIN=ListBuild

all: $(BIN)

$(BIN): $(OBJS)
	g++ -o $(CPPFLAGS) -o $(BIN) $(OBJS) $(LIBS)

clean:
	rm -f *.o *~ $(BIN)
