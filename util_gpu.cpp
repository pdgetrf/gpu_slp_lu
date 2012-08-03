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
	int j_nb = j+nb;
	int j_2nb = j+2*nb;

	int nprow, npcol, myrow, mycol;
    blacs_gridinfo__(&ictxt, &nprow, &npcol, &myrow, &mycol);

	//int mpc = numroc_(&descA[2], &descA[4], &myrow, &izero, &nprow);
	//int nqc = numroc_(&descA[3], &descA[5], &mycol, &izero, &npcol);

	int dmpc = numroc_(&descA2[2], &descA2[4], &myrow, &izero, &nprow);
	int dnqc = numroc_(&descA2[3], &descA2[5], &mycol, &ione, &npcol);
	
	int lda = descA[8];
	int ldda = descA2[8];

	/* 
	 * load data only if I have some trailing matrix to contribute
	 */
	int i_nb = i+nb;
	infog2l_(&i_nb, &j_nb, descA2, &nprow, &npcol, &myrow, &mycol, 
			&diic, &djjc, &icrow, &iccol);
	diic--;	djjc--;
	int dx = dmpc-diic;
	int dy = dnqc-djjc;

	if (dx>0 && dy>0)
	{
#if 0
		infog2l_(&i_nb, &j_2nb, descA, &nprow, &npcol, &myrow, &mycol, &iic, &jjc, &icrow, &iccol);
		iic--;	jjc--;

		printf ("(%d,%d, i=%d, j=%d): dx=%d, dy=%d, iic=%d, jjc=%d\n", myrow, mycol, i, j, dx, dy, iic, jjc); 
		cublasStatus r=	cublasGetMatrix(dx, dy, sizeof(double), dA+djjc*ldda+diic, ldda, &A[jjc*lda+iic], lda);
		if (r!=CUBLAS_STATUS_SUCCESS)
			printf ("cublasStatus error\n");
#endif

		// get pivoting info location from ipiv
		infog2l_(&i, &j, descA, &nprow, &npcol, &myrow, &mycol, 
				&iic, &jjc, &icrow, &iccol);
		int pii = iic-1;

		// copy rows involved in the pivoting to CPU memory
		int ii;
		for (ii=0; ii<nb; ii++)
		{
		//	printf ("(%d,%d, i=%d, j=%d): ipiv[%d] = %d, dx=%d, dy=%d\n", myrow, mycol, i, j, pii+ii, ipiv[pii+ii], dx, dy); 
			infog2l_(&ipiv[pii+ii], &j_2nb, descA, &nprow, &npcol, &myrow, &mycol, &iic, &jjc, &icrow, &iccol);
			iic--;	jjc--;

			if (icrow == myrow)
			{
				infog2l_(&ipiv[pii+ii], &j_nb, descA2, &nprow, &npcol, &myrow, &mycol, &diic, &djjc, &icrow, &iccol);
				diic--;	djjc--;

				cublasStatus r=	cublasGetMatrix(1, dy, sizeof(double), dA+djjc*ldda+diic, ldda, &A[jjc*lda+iic], lda);
				if (r!=CUBLAS_STATUS_SUCCESS)
					printf ("cublasStatus error\n");
			}
		}
	}

	/*
	 * load the next panel
	 */
	infog2l_(&i, &j, descA2, &nprow, &npcol, &myrow, &mycol, 
			&diic, &djjc, &icrow, &iccol);
	diic--;	djjc--;
	dx = dmpc-diic;
	dy = (dnqc-djjc)>0?nb:0;

	if (mycol==iccol && dx>0 && dy>0)
	{
		infog2l_(&i, &j_nb, descA, &nprow, &npcol, &myrow, &mycol, &iic, &jjc, &icrow, &iccol);
		iic--;	jjc--;

		//printf ("(%d,%d, i=%d, j=%d): dx=%d, dy=%d, iic=%d, jjc=%d\n", myrow, mycol, i, j, dx, dy, iic, jjc); 
		cublasStatus r=	cublasGetMatrix(dx, dy, sizeof(double), dA+djjc*ldda+diic, ldda, &A[jjc*lda+iic], lda);
		if (r!=CUBLAS_STATUS_SUCCESS)
			printf ("cublasStatus error\n");
	}

	/*
	 * load the trsm part
	 */
	infog2l_(&i, &j_nb, descA2, &nprow, &npcol, &myrow, &mycol, 
			&diic, &djjc, &icrow, &iccol);
	diic--;	djjc--;
	dx = (dmpc-diic)>0?nb:0;
	dy = dnqc-djjc;

	if (myrow==icrow && dx>0 && dy>0)
	{
		infog2l_(&i, &j_2nb, descA, &nprow, &npcol, &myrow, &mycol, &iic, &jjc, &icrow, &iccol);
		iic--;	jjc--;
		
		cublasStatus r=	cublasGetMatrix(dx, dy, sizeof(double), dA+djjc*ldda+diic, ldda, &A[jjc*lda+iic], lda);
		if (r!=CUBLAS_STATUS_SUCCESS)
			printf ("cublasStatus error\n");
	}

#if 0
	int lda = descA[8];
	int ldda = descA2[8];
	
	if (dx*dy>0)
	{
		/*
		cublasStatus r=	cublasGetMatrix(dx, dy, sizeof(double), dA+djjc*ldda+diic, ldda, &A[jjc*lda+iic], lda);
		if (r!=CUBLAS_STATUS_SUCCESS)
			printf ("cublasStatus error\n");
			*/
		
		cudaMemcpy2DAsync	(&A[jjc*lda+iic], lda*sizeof(double), dA+djjc*ldda+diic, ldda*sizeof(double), 
					 		 dx*sizeof(double), dy,	cudaMemcpyDeviceToHost, fstream);
	}
#endif
}


extern "C"
void Load_all_for_Pivoting (double *A, int i, int j, int *descA, int *ipiv,
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
	int j_nb = j+nb;
	int j_2nb = j+2*nb;

	int nprow, npcol, myrow, mycol;
    blacs_gridinfo__(&ictxt, &nprow, &npcol, &myrow, &mycol);

	//int mpc = numroc_(&descA[2], &descA[4], &myrow, &izero, &nprow);
	//int nqc = numroc_(&descA[3], &descA[5], &mycol, &izero, &npcol);

	int dmpc = numroc_(&descA2[2], &descA2[4], &myrow, &izero, &nprow);
	int dnqc = numroc_(&descA2[3], &descA2[5], &mycol, &ione, &npcol);
	
	int i_nb = i+nb;
	infog2l_(&i_nb, &j_nb, descA2, &nprow, &npcol, &myrow, &mycol, 
			&diic, &djjc, &icrow, &iccol);
	diic--;	djjc--;
	int dx = dmpc-diic;
	int dy = dnqc-djjc;

	int lda = descA[8];
	int ldda = descA2[8];

	// load data only if I have some trailing matrix to contribute
	if (dx>0 && dy>0)
	{
#if 0
		infog2l_(&i_nb, &j_2nb, descA, &nprow, &npcol, &myrow, &mycol, &iic, &jjc, &icrow, &iccol);
		iic--;	jjc--;
		cublasStatus r=	cublasSetMatrix(dx, dy, sizeof(double), &A[jjc*lda+iic], lda, dA+djjc*ldda+diic, ldda);
		if (r!=CUBLAS_STATUS_SUCCESS)
			printf ("cublasStatus error\n");
#endif

		infog2l_(&i, &j, descA, &nprow, &npcol, &myrow, &mycol, 
				&iic, &jjc, &icrow, &iccol);
		int pii = iic-1;

		// copy rows involved in the pivoting to CPU memory
		int ii;
		for (ii=0; ii<nb; ii++)
		{
			infog2l_(&ipiv[pii+ii], &j_2nb, descA, &nprow, &npcol, &myrow, &mycol, &iic, &jjc, &icrow, &iccol);
			iic--;	jjc--;

			if (icrow == myrow)
			{
				//printf ("(%d,%d, i=%d, j=%d): ipiv[%d] = %d, dx=%d, dy=%d\n", myrow, mycol, i, j, ii, ipiv[iic+ii], dx, dy); 
				infog2l_(&ipiv[pii+ii], &j_nb, descA2, &nprow, &npcol, &myrow, &mycol, &diic, &djjc, &icrow, &iccol);
				diic--;	djjc--;

				cublasStatus r=	cublasSetMatrix(1, dy, sizeof(double), &A[jjc*lda+iic], lda, dA+djjc*ldda+diic, ldda);
				if (r!=CUBLAS_STATUS_SUCCESS)
					printf ("cublasStatus error\n");
			}
		}
	}

#if 0
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
		/*
		cublasStatus r=	cublasSetMatrix(dx, dy, sizeof(double), &A[jjc*lda+iic], lda, dA+djjc*ldda+diic, ldda);
		if (r!=CUBLAS_STATUS_SUCCESS)
			printf ("cublasStatus error\n");
			*/
	
		cudaMemcpy2DAsync	(dA+djjc*ldda+diic, ldda*sizeof(double), &A[jjc*lda+iic], lda*sizeof(double),
					 		 dx*sizeof(double), dy,	cudaMemcpyHostToDevice, fstream);
	}
#endif
}

extern "C"
void Save_all_after_Pivoting (double *A, int i, int j, int *descA, int *ipiv,
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
		/*
		cublasStatus r=	cublasSetMatrix(dx, dy, sizeof(double), &A[jjc*lda+iic], lda, dA+djjc*ldda+diic, ldda);
		if (r!=CUBLAS_STATUS_SUCCESS)
			printf ("cublasStatus error\n");
			*/
	
		cudaMemcpy2DAsync	(dA+djjc*ldda+diic, ldda*sizeof(double), &A[jjc*lda+iic], lda*sizeof(double),
					 		 dx*sizeof(double), dy,	cudaMemcpyHostToDevice, fstream);
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

