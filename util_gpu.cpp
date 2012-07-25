#include "util_gpu.h"
#include "stdio.h"

extern "C"
{
	void infog2l_ (int *GRINDX, int *GCINDX, int *DESC, int *NPROW, int *NPCOL, int *MYROW, int *MYCOL, 
			int *LRINDX, int *LCINDX, int *RSRC, int *CSRC );
	int numroc_(int * N, int * NB, int * IPROC, int * ISRCPROC, int * NPROCS );
	extern void blacs_gridinfo__(int *, int *, int *, int *, int *);
}

extern "C"
void Load_for_Pivoting (double *A, int i, int j, int *descA, int *ipiv,
						double *dA, int *descA2, cudaStream_t fstream)
{
	int iic, jjc, icrow, iccol;
	int izero = 0, ione = 1;
    int ictxt = descA[1];
	int nb = descA[4];

	int nprow, npcol, myrow, mycol;
    blacs_gridinfo__(&ictxt, &nprow, &npcol, &myrow, &mycol);

	int x_ = i-nb;
	int y_ = j+nb;
	infog2l_(&x_, &y_, descA2, &nprow, &npcol, &myrow, &mycol, 
			&iic, &jjc, &icrow, &iccol);
	iic--;	jjc--;

	/*
	x_ = descA[2];
	int mpc = numroc_(&i__1, &descC2[4], &myrow, &izero, &nprow);
	x_ = descA[3];
	int nqc = numroc_(&i__1, &descC2[5], &mycol, &ione, &npcol);
	*/


}

extern "C"
void printout_devices( )
{
	int ndevices, idevice;
	cudaGetDeviceCount( &ndevices );

	for( idevice = 0; idevice < 1; idevice++ ) 
	//for( idevice = 0; idevice < ndevices; idevice++ ) 
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties( &prop, idevice );
		printf( "device %d: %s, %.1f MHz clock, %.1f MB memory, capability %d.%d\n",
				idevice,
				prop.name,
				prop.clockRate / 1000.,
				prop.totalGlobalMem / (1024.*1024.),
				prop.major,
				prop.minor );
	}
}

