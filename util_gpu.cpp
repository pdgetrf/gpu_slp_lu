#include "util_gpu.h"
#include "stdio.h"
#include "string.h"

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
	int diic, djjc, iic, jjc, icrow, iccol;
	int izero = 0, ione = 1;
    int ictxt = descA[1];
	int nb = descA[4];

	int nprow, npcol, myrow, mycol;
    blacs_gridinfo__(&ictxt, &nprow, &npcol, &myrow, &mycol);

	//int mpc = numroc_(&descA[2], &descA[4], &myrow, &izero, &nprow);
	//int nqc = numroc_(&descA[3], &descA[5], &mycol, &izero, &npcol);

	int dmpc = numroc_(&descA2[2], &descA2[4], &myrow, &izero, &nprow);
	int dnqc = numroc_(&descA2[3], &descA2[5], &mycol, &ione, &npcol);
	
	infog2l_(&i, &j, descA, &nprow, &npcol, &myrow, &mycol, 
			&iic, &jjc, &icrow, &iccol);
	iic--;	jjc--;
	//int xx = mpc-iic;
	//int yy = nqc-jjc;

	int j_nb = j-nb;
	infog2l_(&i, &j_nb, descA2, &nprow, &npcol, &myrow, &mycol, 
			&diic, &djjc, &icrow, &iccol);
	diic--;	djjc--;
	int dx = dmpc-diic;
	int dy = dnqc-djjc;

	int lda = descA[8];
	int ldda = descA2[8];
	
	if (dx*dy>0)
	{
		cublasStatus r=	cublasGetMatrix(dx, dy, sizeof(double), dA+djjc*ldda+diic, ldda, &A[jjc*lda+iic], lda);
		if (r!=CUBLAS_STATUS_SUCCESS)
			printf ("cublasStatus error\n");
		
		/*
		cudaMemcpy2DAsync	(&A[jjc*lda+iic], lda*sizeof(double), dA+djjc*ldda+diic, ldda*sizeof(double), 
					 		 dx*sizeof(double), dy,	cudaMemcpyDeviceToHost, fstream);
			*/
	}
}

extern "C"
void Save_after_Pivoting (double *A, int i, int j, int *descA, int *ipiv,
						double *dA, int *descA2, cudaStream_t fstream)
{
	int diic, djjc, iic, jjc, icrow, iccol;
	int izero = 0, ione = 1;
    int ictxt = descA[1];
	int nb = descA[4];

	int nprow, npcol, myrow, mycol;
    blacs_gridinfo__(&ictxt, &nprow, &npcol, &myrow, &mycol);

	//int mpc = numroc_(&descA[2], &descA[4], &myrow, &izero, &nprow);
	//int nqc = numroc_(&descA[3], &descA[5], &mycol, &izero, &npcol);

	int dmpc = numroc_(&descA2[2], &descA2[4], &myrow, &izero, &nprow);
	int dnqc = numroc_(&descA2[3], &descA2[5], &mycol, &ione, &npcol);
	
	infog2l_(&i, &j, descA, &nprow, &npcol, &myrow, &mycol, 
			&iic, &jjc, &icrow, &iccol);
	iic--;	jjc--;
	//int xx = mpc-iic;
	//int yy = nqc-jjc;

	int j_nb=j-nb;
	infog2l_(&i, &j_nb, descA2, &nprow, &npcol, &myrow, &mycol, 
			&diic, &djjc, &icrow, &iccol);
	diic--;	djjc--;
	int dx = dmpc-diic;
	int dy = dnqc-djjc;

	int lda = descA[8];
	int ldda = descA2[8];

//	printf ("(%d,%d): saving %d x %d from C(%d,%d) to G(%d/%d) when i=%d, j=%d, ldda=%d\n", myrow, mycol, dx, dy, iic, jjc, diic, djjc, i, j, ldda);
	if (dx*dy>0)
	{
		cublasStatus r=	cublasSetMatrix(dx, dy, sizeof(double), &A[jjc*lda+iic], lda, dA+djjc*ldda+diic, ldda);
		if (r!=CUBLAS_STATUS_SUCCESS)
			printf ("cublasStatus error\n");
	
		/*
		cudaMemcpy2DAsync	(dA+djjc*ldda+diic, ldda*sizeof(double), &A[jjc*lda+iic], lda*sizeof(double),
					 		 dx*sizeof(double), dy,	cudaMemcpyHostToDevice, fstream);
			*/
	}
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

