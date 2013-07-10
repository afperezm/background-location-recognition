# Top-level Makefile

all: default

default:
# Make libraries
	cd ../VocabTree2/VocabLib; $(MAKE)
# Make program
	cd Common; $(MAKE)
	cd DataLib; $(MAKE)
	cd FeatureExtractSelect; $(MAKE)
	cd GeomVerify; $(MAKE)
	cd ListBuild; $(MAKE)
	cd PerfEval; $(MAKE)

clean:
#	cd ../VocabTree2/VocabLib; $(MAKE) clean
	cd Common; $(MAKE) clean
	cd DataLib; $(MAKE) clean
	cd FeatureExtractSelect; $(MAKE) clean
	cd GeomVerify; $(MAKE) clean
	cd ListBuild; $(MAKE) clean
	cd PerfEval; $(MAKE) clean
