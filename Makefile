

CC := nvcc
SRCDIR := src
BUILDDIR := build
TARGETDIR := bin/$(DSET)/$(KCHI)
#TARGETDIR := bin/$(DSET)
TARGET := $(TARGETDIR)/XSMD.so

DATA := data/$(DSET)
SRCEXT := cu
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))

DATAPARAMS := $(DATA)/mol_param.cu $(DATA)/env_param.cu 
SCATPARAMS := $(DATA)/scat_param.cu
REFPARAMS := $(DATA)/coord_ref.cu

PARAMS := $(SRCDIR)/WaasKirf.cu $(SRCDIR)/XSMD.cu
FITPARAMS := $(SRCDIR)/WaasKirf.cu $(SRCDIR)/fit_initial.cu
FITTRAJPARAMS := $(SRCDIR)/WaasKirf.cu $(SRCDIR)/fit_traj_initial.cu
TRAJPARAMS := $(SRCDIR)/WaasKirf.cu $(SRCDIR)/traj_scatter.cu
TESTPARAMS := $(SRCDIR)/WaasKirf.cu $(SRCDIR)/speedtest.cu

SOURCESOBJ := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(SOURCES:.$(SRCEXT)=.o))
PARAMSOBJ := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(PARAMS:.$(SRCEXT)=.o))
FITPARAMSOBJ := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(FITPARAMS:.$(SRCEXT)=.o))
FITTRAJPARAMSOBJ := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(FITTRAJPARAMS:.$(SRCEXT)=.o))
TRAJPARAMSOBJ := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(TRAJPARAMS:.$(SRCEXT)=.o))
TESTPARAMSOBJ := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(TESTPARAMS:.$(SRCEXT)=.o))
DATAPARAMSOBJ := $(patsubst $(DATA)/%, $(BUILDDIR)/%, $(DATAPARAMS:.$(SRCEXT)=.o))
SCATPARAMSOBJ := $(patsubst $(DATA)/%, $(BUILDDIR)/%, $(SCATPARAMS:.$(SRCEXT)=.o))
REFPARAMSOBJ := $(patsubst $(DATA)/%, $(BUILDDIR)/%, $(REFPARAMS:.$(SRCEXT)=.o))

CFLAGS := --compiler-options='-fPIC' -use_fast_math -lineinfo --ptxas-options=-v
LIB := -lgsl -lgslcblas -lm
INC := -Iinclude -Idata/$(DSET)
GSLINC := #-I/YOUR_GSL_INCLUDE
GSLLIB := #-L/YOUR_GSL_LIB

#ifndef DSET
#$(error DSET is not set. DSET points to the directory e.g. DSET=1aki if data is in /data/1aki/)
#endif


$(TARGET): $(BUILDDIR)/XSMD_wrap.o $(PARAMSOBJ) $(DATAPARAMSOBJ) $(SCATPARAMSOBJ)
	@mkdir -p bin/$(DSET)/$(KCHI)
	@mkdir -p bin/$(DSET)/$(KCHI)/backup_code
	$(CC) -shared $^ -o $(TARGET)
	@cp $(SRCDIR)/*.cu include/*.hh $(TARGETDIR)/backup_code
	#nvcc --compiler-options='-fPIC' -use_fast_math -lineinfo --ptxas-options=-v -c XSMD.cu mol_param.cu scat_param.cu env_param.cu WaasKirf.cu XSMD_wrap.cxx 
	#nvcc -shared XSMD.o mol_param.o scat_param.o env_param.o WaasKirf.o XSMD_wrap.o -o XSMD.so
test: $(BUILDDIR)/speedtest.o $(TESTPARAMSOBJ) $(DATAPARAMSOBJ) $(SCATPARAMSOBJ) $(REFPARAMSOBJ)
	@mkdir -p bin/$(DSET)
	@echo "Linking for speed test ......"
	$(CC) $(GSLLIB) $(LIB) $^ -o bin/$(DSET)/speedtest.out
traj: $(BUILDDIR)/traj_scatter.o  
	@mkdir -p bin/
	@echo "Linking for traj to scatter ......"
	$(CC) $(GSLLIB) $(LIB) $^ -o bin/$(DSET)/traj_scatter.out
initial: $(BUILDDIR)/structure_calc.o $(DATAPARAMSOBJ)
	$(CC) $(CFLAGS) $^ -o bin/$(DSET)/structure_calc.out
fit: $(BUILDDIR)/fit_initial.o $(FITPARAMSOBJ) $(DATAPARAMSOBJ) $(DATA)/coord_ref.cu $(DATA)/expt_data.cu
	@echo "Linking for fit ......"
	@mkdir -p bin/$(DSET)
	$(CC) $(GSLLIB) $^ $(LIB) -o bin/$(DSET)/fit_initial.out
fit_traj: $(BUILDDIR)/fit_traj_initial.o $(FITTRAJPARAMSOBJ) $(DATAPARAMSOBJ) $(DATA)/expt_data.cu
	@echo "Linking for fit trajectory ......"
	@mkdir -p bin/$(DSET)
	$(CC) $(GSLLIB) $^ $(LIB) -o bin/$(DSET)/fit_traj_initial.out
$(BUILDDIR)/XSMD_wrap.o: $(SRCDIR)/XSMD_wrap.cxx
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) $(GSLINC) $(LIB) -c -o $@ $^
$(SRCDIR)/XSMD_wrap.cxx: $(SRCDIR)/XSMD.i
	swig -c++ -tcl $(INC) $^
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) $(GSLINC) $(LIB) -c -o $@ $^
$(BUILDDIR)/%.o: $(DATA)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) $(GSLINC) $(LIB) -c -o $@ $^

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)	
	@echo " $(RM) -r $(SRCDIR)/XSMD_wrap.cxx"; $(RM) -r $(SRCDIR)/XSMD_wrap.cxx

