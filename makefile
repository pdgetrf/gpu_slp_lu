#	makefile for high-level threading LU	#

CC = mpicc 
FC = mpif77 

HOME = /home/du

BLACSLIB =  $(HOME)/lib/BLACS/LIB/blacsCinit_MPI-LINUX-0.a  $(HOME)/lib/BLACS/LIB/blacsF77init_MPI-LINUX-0.a  $(HOME)/lib/BLACS/LIB/blacs_MPI-LINUX-0.a

MKLROOT   = /mnt/scratch/sw/intel/2011.2.137/mkl
FASTBLASLIB = -L$(MKLROOT)/lib/intel64 -Wl,--start-group -lscalapack $(BLACSLIB)  -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group -fopenmp
#FASTBLASLIB = -L$(MKLROOT)/lib/intel64 -L$(SLPPATH) -Wl,--start-group -lmkl_scalapack_lp64 -lmkl_blacs_lp64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group -fopenmp
GPULIB = -L/opt/cuda/lib64 -lcublas -lcudart -lcuda

CUDADIR   = /mnt/scratch/cuda
INC       = -I$(CUDADIR)/include 


CFLAGS = -I/opt/mkl/include -I/opt/cuda/include -Wall -I/home/du/lib/scalapack-1.8.0/PBLAS/SRC -I/home/du/lib/scalapack-1.8.0/PBLAS/SRC/PTOOLS -fopenmp
#CFLAGS += -DTIMING
FFLAGS =  -fsecond-underscore

LDFLAGS =  -L$(HOME)/lib/lapack-3.3.1\
		   -L$(HOME)/lib/scalapack-1.8.0 \
		   -L$(HOME)/lib/lapack-3.3.0 \
		   -L$(HOME)/lib/BLACS/LIB 


LINKLIB =   lu_test.o pdgeqrrv.o pdmatgen.o pmatgeninc.o pdlaprnt.o orig_pdgetrf.o gpu_pdgeqrf.o gpu_pdlarfb.o util_ft.o util_gpu.o pdgetrrv.o \
			gpu_pdgetrf.o pdgemm_.o PB_CpgemmAB.o pdtrsm_.o PB_CptrsmAB.o PB_CptrsmAB0.o $(FASTBLASLIB) $(GPULIB)

PROG = lu_test.x

$(PROG) : lu_test.o pdgeqrrv.o pdmatgen.o pmatgeninc.o pdlaprnt.o util_ft.o orig_pdgetrf.o gpu_pdgeqrf.o gpu_pdlarfb.o util_gpu.o pdgetrrv.o gpu_pdgetrf.o pdgemm_.o PB_CpgemmAB.o pdtrsm_.o PB_CptrsmAB.o PB_CptrsmAB0.o
	$(FC) -o $(PROG) $(CFLAGS) $(LDFLAGS) $(LINKLIB) 

.c.o:
	$(CC) -c -O3 $(DEBUG) $(CFLAGS)  $*.c

.cpp.o:
	$(CC) -c -O3 $(DEBUG) $(CFLAGS)  $*.cpp

.f.o:
	$(FC) -c -O3 $(DEBUG) $(FFLAGS)  $*.f 

clean:
	rm *.o *.x
