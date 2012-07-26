/* pdgetrf.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "f2c.h"
#include "util_gpu.h"
#include "mpi.h"

extern void infog2l_ (int *GRINDX, int *GCINDX, int *DESC, int *NPROW, int *NPCOL, int *MYROW, int *MYCOL, 
					   int *LRINDX, int *LCINDX, int *RSRC, int *CSRC );
extern int numroc_(int * N, int * NB, int * IPROC, int * ISRCPROC, int * NPROCS );


double *dA, *dB, *dC;

#define TIMING
#ifdef TIMING
  double t1 = 0.0f; double t2 = 0.0f; double tswap = 0.0f; double tload = 0.0f; double tsave = 0.0f; double ttrsm = 0.0f; 
# define TIME(tt, TEXT ) do { t1 = MPI_Wtime(); TEXT ; t2 = MPI_Wtime(); tt += t2-t1; } while(0)
#else
# define TIME(tt, TEXT) TEXT
#endif



/* Table of constant values */

static int c__1 = 1;
static int c__2 = 2;
static int c__6 = 6;
static int c__0 = 0;
static double c_b31 = 1.;
static double c_b34 = -1.;
static int c_n1 = -1;

/* Subroutine */ int gpu_pdgetrf_(int *m, int *n, double *a, int *
	ia, int *ja, int *desca, int *ipiv, int *info)
{
    /* System generated locals */
    int i__1, i__2, i__3, i__4, i__5, i__6, i__7, i__8;

    /* Local variables */
    static int i__, j;
    extern /* Subroutine */ int pb_topget__(int *, char *, char *, char *,
	     ftnlen, ftnlen, ftnlen), pb_topset__(int *, char *, char *, 
	    char *, ftnlen, ftnlen, ftnlen);
    static int jb, in, jn, mn, idum1[1], idum2[1];
    extern int iceil_(int *, int *);
    static int icoff, iinfo, iroff, npcol, mycol, ictxt, nprow, myrow;
    extern /* Subroutine */ int gpu_pdgemm_(char *, char *, int *, int *, 
	    int *, double *, double *, int *, int *, 
	    int *, double *, int *, int *, int *, 
	    double *, double *, int *, int *, int *, int *, double *), 
	    blacs_gridinfo__(int *, int *, int *,
	     int *, int *), gpu_pdtrsm_(char *, char *, char *, char *, 
	    int *, int *, double *, double *, int *, 
	    int *, int *, double *, int *, int *, int 
	    *, ftnlen, ftnlen, ftnlen, ftnlen), igamn2d_(int *, char *, 
	    char *, int *, int *, int *, int *, int *, 
	    int *, int *, int *, int *, ftnlen, ftnlen), 
	    chk1mat_(int *, int *, int *, int *, int *, 
	    int *, int *, int *, int *), pdgetf2_(int *, 
	    int *, double *, int *, int *, int *, int 
	    *, int *), pxerbla_(int *, char *, int *, ftnlen);
    static char colbtop[1], colctop[1];
    extern /* Subroutine */ int pdlaswp_(char *, char *, int *, 
	    double *, int *, int *, int *, int *, int 
	    *, int *, ftnlen, ftnlen);
    static char rowbtop[1];
    extern /* Subroutine */ int pchk1mat_(int *, int *, int *, 
	    int *, int *, int *, int *, int *, int *, 
	    int *, int *, int *);


/*  -- ScaLAPACK routine (version 1.7) -- */
/*     University of Tennessee, Knoxville, Oak Ridge National Laboratory, */
/*     and University of California, Berkeley. */
/*     May 25, 2001 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  PDGETRF computes an LU factorization of a general M-by-N distributed */
/*  matrix sub( A ) = (IA:IA+M-1,JA:JA+N-1) using partial pivoting with */
/*  row interchanges. */

/*  The factorization has the form sub( A ) = P * L * U, where P is a */
/*  permutation matrix, L is lower triangular with unit diagonal ele- */
/*  ments (lower trapezoidal if m > n), and U is upper triangular */
/*  (upper trapezoidal if m < n). L and U are stored in sub( A ). */

/*  This is the right-looking Parallel Level 3 BLAS version of the */
/*  algorithm. */

/*  Notes */
/*  ===== */

/*  Each global data object is described by an associated description */
/*  vector.  This vector stores the information required to establish */
/*  the mapping between an object element and its corresponding process */
/*  and memory location. */

/*  Let A be a generic term for any 2D block cyclicly distributed array. */
/*  Such a global array has an associated description vector DESCA. */
/*  In the following comments, the character _ should be read as */
/*  "of the global array". */

/*  NOTATION        STORED IN      EXPLANATION */
/*  --------------- -------------- -------------------------------------- */
/*  DTYPE_A(global) DESCA( DTYPE_ )The descriptor type.  In this case, */
/*                                 DTYPE_A = 1. */
/*  CTXT_A (global) DESCA( CTXT_ ) The BLACS context handle, indicating */
/*                                 the BLACS process grid A is distribu- */
/*                                 ted over. The context itself is glo- */
/*                                 bal, but the handle (the int */
/*                                 value) may vary. */
/*  M_A    (global) DESCA( M_ )    The number of rows in the global */
/*                                 array A. */
/*  N_A    (global) DESCA( N_ )    The number of columns in the global */
/*                                 array A. */
/*  MB_A   (global) DESCA( MB_ )   The blocking factor used to distribute */
/*                                 the rows of the array. */
/*  NB_A   (global) DESCA( NB_ )   The blocking factor used to distribute */
/*                                 the columns of the array. */
/*  RSRC_A (global) DESCA( RSRC_ ) The process row over which the first */
/*                                 row of the array A is distributed. */
/*  CSRC_A (global) DESCA( CSRC_ ) The process column over which the */
/*                                 first column of the array A is */
/*                                 distributed. */
/*  LLD_A  (local)  DESCA( LLD_ )  The leading dimension of the local */
/*                                 array.  LLD_A >= MAX(1,LOCr(M_A)). */

/*  Let K be the number of rows or columns of a distributed matrix, */
/*  and assume that its process grid has dimension p x q. */
/*  LOCr( K ) denotes the number of elements of K that a process */
/*  would receive if K were distributed over the p processes of its */
/*  process column. */
/*  Similarly, LOCc( K ) denotes the number of elements of K that a */
/*  process would receive if K were distributed over the q processes of */
/*  its process row. */
/*  The values of LOCr() and LOCc() may be determined via a call to the */
/*  ScaLAPACK tool function, NUMROC: */
/*          LOCr( M ) = NUMROC( M, MB_A, MYROW, RSRC_A, NPROW ), */
/*          LOCc( N ) = NUMROC( N, NB_A, MYCOL, CSRC_A, NPCOL ). */
/*  An upper bound for these quantities may be computed by: */
/*          LOCr( M ) <= ceil( ceil(M/MB_A)/NPROW )*MB_A */
/*          LOCc( N ) <= ceil( ceil(N/NB_A)/NPCOL )*NB_A */

/*  This routine requires square block decomposition ( MB_A = NB_A ). */

/*  Arguments */
/*  ========= */

/*  M       (global input) INTEGER */
/*          The number of rows to be operated on, i.e. the number of rows */
/*          of the distributed submatrix sub( A ). M >= 0. */

/*  N       (global input) INTEGER */
/*          The number of columns to be operated on, i.e. the number of */
/*          columns of the distributed submatrix sub( A ). N >= 0. */

/*  A       (local input/local output) DOUBLE PRECISION pointer into the */
/*          local memory to an array of dimension (LLD_A, LOCc(JA+N-1)). */
/*          On entry, this array contains the local pieces of the M-by-N */
/*          distributed matrix sub( A ) to be factored. On exit, this */
/*          array contains the local pieces of the factors L and U from */
/*          the factorization sub( A ) = P*L*U; the unit diagonal ele- */
/*          ments of L are not stored. */

/*  IA      (global input) INTEGER */
/*          The row index in the global array A indicating the first */
/*          row of sub( A ). */

/*  JA      (global input) INTEGER */
/*          The column index in the global array A indicating the */
/*          first column of sub( A ). */

/*  DESCA   (global and local input) INTEGER array of dimension DLEN_. */
/*          The array descriptor for the distributed matrix A. */

/*  IPIV    (local output) INTEGER array, dimension ( LOCr(M_A)+MB_A ) */
/*          This array contains the pivoting information. */
/*          IPIV(i) -> The global row local row i was swapped with. */
/*          This array is tied to the distributed matrix A. */

/*  INFO    (global output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  If the i-th argument is an array and the j-entry had */
/*                an illegal value, then INFO = -(i*100+j), if the i-th */
/*                argument is a scalar and had an illegal value, then */
/*                INFO = -i. */
/*          > 0:  If INFO = K, U(IA+K-1,JA+K-1) is exactly zero. */
/*                The factorization has been completed, but the factor U */
/*                is exactly singular, and division by zero will occur if */
/*                it is used to solve a system of equations. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */
	static cudaStream_t fstream;
	cudaStreamCreate(&fstream);

/*     Get grid parameters */

    /* Parameter adjustments */
    --ipiv;
    --desca;
    --a;

    /* Function Body */
    ictxt = desca[2];
    blacs_gridinfo__(&ictxt, &nprow, &npcol, &myrow, &mycol);

/*     Test the input parameters */

    *info = 0;
    if (nprow == -1) {
	*info = -602;
    } else {
	chk1mat_(m, &c__1, n, &c__2, ia, ja, &desca[1], &c__6, info);
	if (*info == 0) {
	    iroff = (*ia - 1) % desca[5];
	    icoff = (*ja - 1) % desca[6];
	    if (iroff != 0) {
		*info = -4;
	    } else if (icoff != 0) {
		*info = -5;
	    } else if (desca[5] != desca[6]) {
		*info = -606;
	    }
	}
	pchk1mat_(m, &c__1, n, &c__2, ia, ja, &desca[1], &c__6, &c__0, idum1, 
		idum2, info);
    }

    if (*info != 0) {
	i__1 = -(*info);
	pxerbla_(&ictxt, "PDGETRF", &i__1, (ftnlen)7);
	return 0;
    }

/*     Quick return if possible */

    if (desca[3] == 1) {
	ipiv[1] = 1;
	return 0;
    } else if (*m == 0 || *n == 0) {
	return 0;
    }

/*     Split-ring topology for the communication along process rows */

    pb_topget__(&ictxt, "Broadcast", "Rowwise", rowbtop, (ftnlen)9, (ftnlen)7,
	     (ftnlen)1);
    pb_topget__(&ictxt, "Broadcast", "Columnwise", colbtop, (ftnlen)9, (
	    ftnlen)10, (ftnlen)1);
    pb_topget__(&ictxt, "Combine", "Columnwise", colctop, (ftnlen)7, (ftnlen)
	    10, (ftnlen)1);
    pb_topset__(&ictxt, "Broadcast", "Rowwise", "S-ring", (ftnlen)9, (ftnlen)
	    7, (ftnlen)6);
    pb_topset__(&ictxt, "Broadcast", "Columnwise", " ", (ftnlen)9, (ftnlen)10,
	     (ftnlen)1);
    pb_topset__(&ictxt, "Combine", "Columnwise", " ", (ftnlen)7, (ftnlen)10, (
	    ftnlen)1);
	
	// allocate memory on GPU for V and A2 and W
	int nb = desca[5];
	int ic, jc, iic, jjc, icrow, iccol; 
	int lda = desca[9];
	ic = 1;
	jc = 1 + desca[5];
	infog2l_(&ic, &jc, &desca[1], &nprow, &npcol, &myrow, &mycol, 
									&iic, &jjc, &icrow, &iccol);
	iic--;	jjc--;
	
	int descC2[9];
	int n_A2 = *n - desca[5];
	int izero = 0, ione = 1;
	descinit_ (descC2, m, &n_A2, &desca[5], &desca[5], &izero, &ione, &ictxt, &lda, &iinfo);
	if (iinfo!=0)
	{
		printf ("error at descinit_, %d, %s\n", __LINE__, __FILE__);
		exit(0);
	}

	i__1 = *m;
	int mpc = numroc_(&i__1, &descC2[4], &myrow, &izero, &nprow);
	i__1 = *n-descC2[4];
	int nqc = numroc_(&i__1, &descC2[5], &mycol, &ione, &npcol);
	
	double *pinnbuf=NULL;
	if (mpc*nqc>0)
	{
		//printf ("\n(%d,%d) mpc=%d, nqc=%d, ldc=%d\n", myrow, mycol, mpc, nqc, lda);
		TESTING_DEVALLOC (dC, double, mpc*nqc);
		TESTING_DEVALLOC (dB, double, nqc*nb);
		TESTING_DEVALLOC (dA, double, mpc*nb);
//		TESTING_HOSTALLOC(pinnbuf, double, mpc*descC2[5]);

	}

/*     Handle the first block of columns separately */

    mn = min(*m,*n);
/* Computing MIN */
    i__1 = iceil_(ia, &desca[5]) * desca[5], i__2 = *ia + *m - 1;
    in = min(i__1,i__2);
/* Computing MIN */
    i__1 = iceil_(ja, &desca[6]) * desca[6], i__2 = *ja + mn - 1;
    jn = min(i__1,i__2);
    jb = jn - *ja + 1;

/*     Factor diagonal and subdiagonal blocks and test for exact */
/*     singularity. */

    pdgetf2_(m, &jb, &a[1], ia, ja, &desca[1], &ipiv[1], info);

    if (jb + 1 <= *n) 
	{

		/*        Apply interchanges to columns JN+1:JA+N-1. */

		i__1 = *n - jb;
		i__2 = jn + 1;
		pdlaswp_("Forward", "Rows", &i__1, &a[1], ia, &i__2, &desca[1], ia, &
				in, &ipiv[1], (ftnlen)7, (ftnlen)4);

		/*        Compute block row of U. */

		TIME(tsave,
		Save_after_Pivoting (&a[1], *ia+nb, *ja+2*nb, &desca[1], &ipiv[1], dC, descC2, fstream);
		);

		i__1 = *n - jb;
		i__2 = jn + 1;
		TIME(ttrsm,
		gpu_pdtrsm_("Left", "Lower", "No transpose", "Unit", &jb, &i__1, &c_b31, &
				a[1], ia, ja, &desca[1], &a[1], ia, &i__2, &desca[1], (ftnlen)
				4, (ftnlen)5, (ftnlen)12, (ftnlen)4);
		);
		
		cudaStreamSynchronize	(fstream); 	

		if (jb + 1 <= *m) 
		{
			/*           Update trailing submatrix. */

			i__1 = *m - jb;
			i__2 = *n - jb;
			i__3 = in + 1;
			i__4 = jn + 1;
			i__5 = in + 1;
			i__6 = jn + 1;

			// xxx
			gpu_pdgemm_("No transpose", "No transpose", &i__1, &i__2, &jb, &c_b34,
					&a[1], &i__3, ja, &desca[1], &a[1], ia, &i__4, &desca[1],
					&c_b31, &a[1], &i__5, &i__6, &desca[1], descC2, pinnbuf);
		}
	}

/*     Loop over the remaining blocks of columns. */

    i__1 = *ja + mn - 1;
    i__2 = desca[6];
    for (j = jn + 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) 
	{
		/* Computing MIN */
		i__3 = mn - j + *ja;
		jb = min(i__3,desca[6]);
		i__ = *ia + j - *ja;
		
		if (j - *ja + jb + 1 <= *n) 
		{
			TIME(tload,
			Load_for_Pivoting (&a[1], i__, j+nb, &desca[1], &ipiv[1], dC, descC2, fstream);
			);
		}

		/*        Factor diagonal and subdiagonal blocks and test for exact */
		/*        singularity. */

		i__3 = *m - j + *ja;
		pdgetf2_(&i__3, &jb, &a[1], &i__, &j, &desca[1], &ipiv[1], &iinfo);

		if (*info == 0 && iinfo > 0) {
			*info = iinfo + j - *ja;
		}

		/*        Apply interchanges to columns JA:J-JA. */
		

		i__3 = j - *ja;
		i__4 = i__ + jb - 1;
		
		TIME(tswap, 
		pdlaswp_("Forward", "Rowwise", &i__3, &a[1], ia, ja, &desca[1], &i__, 
				&i__4, &ipiv[1], (ftnlen)7, (ftnlen)7);
			);
			
		cudaStreamSynchronize	(fstream); 	
		
		if (j - *ja + jb + 1 <= *n) 
		{

			/*           Apply interchanges to columns J+JB:JA+N-1. */

			i__3 = *n - j - jb + *ja;
			i__4 = j + jb;
			i__5 = i__ + jb - 1;

			pdlaswp_("Forward", "Rowwise", &i__3, &a[1], ia, &i__4, &desca[1],
					&i__, &i__5, &ipiv[1], (ftnlen)7, (ftnlen)7);

			TIME(tsave, 
			Save_after_Pivoting (&a[1], i__+nb, j+2*nb, &desca[1], &ipiv[1], dC, descC2, fstream);
			);

			/*           Compute block row of U. */

			i__3 = *n - j - jb + *ja;
			i__4 = j + jb;
			TIME(ttrsm,
			gpu_pdtrsm_("Left", "Lower", "No transpose", "Unit", &jb, &i__3, &
					c_b31, &a[1], &i__, &j, &desca[1], &a[1], &i__, &i__4, &
					desca[1], (ftnlen)4, (ftnlen)5, (ftnlen)12, (ftnlen)4);
			);
		
			cudaStreamSynchronize	(fstream); 	
			

			if (j - *ja + jb + 1 <= *m) 
			{
				/*              Update trailing submatrix. */

				i__3 = *m - j - jb + *ja;
				i__4 = *n - j - jb + *ja;
				i__5 = i__ + jb;
				i__6 = j + jb;
				i__7 = i__ + jb;
				i__8 = j + jb;
				gpu_pdgemm_("No transpose", "No transpose", &i__3, &i__4, &jb, &
						c_b34, &a[1], &i__5, &j, &desca[1], &a[1], &i__, &
						i__6, &desca[1], &c_b31, &a[1], &i__7, &i__8, &desca[1], descC2, pinnbuf);

			}
		}
		/* L10: */
	}

    if (*info == 0) {
	*info = mn + 1;
    }
    igamn2d_(&ictxt, "Rowwise", " ", &c__1, &c__1, info, &c__1, idum1, idum2, 
	    &c_n1, &c_n1, &mycol, (ftnlen)7, (ftnlen)1);
    if (*info == mn + 1) {
	*info = 0;
    }

    pb_topset__(&ictxt, "Broadcast", "Rowwise", rowbtop, (ftnlen)9, (ftnlen)7,
	     (ftnlen)1);
    pb_topset__(&ictxt, "Broadcast", "Columnwise", colbtop, (ftnlen)9, (
	    ftnlen)10, (ftnlen)1);
    pb_topset__(&ictxt, "Combine", "Columnwise", colctop, (ftnlen)7, (ftnlen)
	    10, (ftnlen)1);
	
	/* free gpu memory */
	if (mpc*nqc>0)
	{
		//TESTING_HOSTFREE(pinnbuf);
		TESTING_DEVFREE(dA);
		TESTING_DEVFREE(dB);
		TESTING_DEVFREE(dC);
	}
	cudaStreamDestroy(fstream);

	if (myrow==0 && mycol==0)
		printf ("tload = %f, tswap =%f || tsave = %f, ttrsm = %f\n", tload, tswap, tsave, ttrsm);

    return 0;

/*     End of PDGETRF */

} /* pdgetrf_ */

