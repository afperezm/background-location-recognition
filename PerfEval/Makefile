# Makefile for PerfEval

OPTFLAGS=-O2
OTHERFLAGS=-g -Wall -fmessage-length=0 -std=c++0x

INCLUDE_PATH=-I../ -I../../VocabTree2
LIB_PATH=-L../../VocabTree2 -L/usr/include -L../

OBJS=PerfEval.o

LIBS=-lcommon -lreader -lvocab `pkg-config opencv --libs`

CPPFLAGS=$(INCLUDE_PATH) $(LIB_PATH) $(OTHERFLAGS) $(OPTFLAGS)

BIN=PerfEval

all: $(BIN)

$(BIN): $(OBJS)
	g++ -o $(CPPFLAGS) -o $(BIN) $(OBJS) $(LIBS)

clean:
	rm -f *.o *~ $(BIN)
