/* orig_pdgeqrf.f -- translated by f2c (version 20061008).
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
#include "f2c.h"
#include "util_gpu.h"
#include "slp.h"

/* Table of constant values */

static int c__1 = 1;
static int c__2 = 2;
static int c__6 = 6;

/* Subroutine */ int gpu_pdgeqrf_(int *m, int *n, doublereal *a, 
	int *ia, int *ja, int *desca, doublereal *tau, doublereal 
	*work, int *lwork, int *info)
{
    /* System generated locals */
    int i__1, i__2, i__3, i__4, i__5;

    /* Local variables */
    static int i__, j, k;
    extern /* Subroutine */ int pb_topget__(int *, char *, char *, char *,
	     ftnlen, ftnlen, ftnlen), pb_topset__(int *, char *, char *, 
	    char *, ftnlen, ftnlen, ftnlen);
    static int jb, jn, mp0, nq0, ipw, idum1[1], idum2[1];
    extern int iceil_(int *, int *);
    static int icoff, iacol, iinfo, npcol, iarow, mycol, lwmin, ictxt, 
	    nprow, myrow;
    extern /* Subroutine */ int blacs_gridinfo__(int *, int *, 
	    int *, int *, int *);
    extern int numroc_(int *, int *, int *, int *, 
	    int *);
    static logical lquery;
    extern /* Subroutine */ int chk1mat_(int *, int *, int *, 
	    int *, int *, int *, int *, int *, int *),
	     pdgeqr2_(int *, int *, doublereal *, int *, int *
	    , int *, doublereal *, doublereal *, int *, int *);
    extern int indxg2p_(int *, int *, int *, int *, 
	    int *);
    extern /* Subroutine */ int gpu_pdlarfb_(char *, char *, char *, char *, 
	    int *, int *, int *, doublereal *, int *, int 
	    *, int *, doublereal *, doublereal *, int *, int *, 
	    int *, doublereal *, ftnlen, ftnlen, ftnlen, ftnlen, 
		double *, int *, double *, double *, double *), 
	    pdlarft_(char *, char *, int *, int *, doublereal *, 
	    int *, int *, int *, doublereal *, doublereal *, 
	    doublereal *, ftnlen, ftnlen), pxerbla_(int *, char *, 
	    int *, ftnlen);
    static char colbtop[1], rowbtop[1];
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

/*  PDGEQRF computes a QR factorization of a real distributed M-by-N */
/*  matrix sub( A ) = A(IA:IA+M-1,JA:JA+N-1) = Q * R. */

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
/*          On entry, the local pieces of the M-by-N distributed matrix */
/*          sub( A ) which is to be factored.  On exit, the elements on */
/*          and above the diagonal of sub( A ) contain the min(M,N) by N */
/*          upper trapezoidal matrix R (R is upper triangular if M >= N); */
/*          the elements below the diagonal, with the array TAU, */
/*          represent the orthogonal matrix Q as a product of elementary */
/*          reflectors (see Further Details). */

/*  IA      (global input) INTEGER */
/*          The row index in the global array A indicating the first */
/*          row of sub( A ). */

/*  JA      (global input) INTEGER */
/*          The column index in the global array A indicating the */
/*          first column of sub( A ). */

/*  DESCA   (global and local input) INTEGER array of dimension DLEN_. */
/*          The array descriptor for the distributed matrix A. */

/*  TAU     (local output) DOUBLE PRECISION array, dimension */
/*          LOCc(JA+MIN(M,N)-1). This array contains the scalar factors */
/*          TAU of the elementary reflectors. TAU is tied to the */
/*          distributed matrix A. */

/*  WORK    (local workspace/local output) DOUBLE PRECISION array, */
/*                                                     dimension (LWORK) */
/*          On exit, WORK(1) returns the minimal and optimal LWORK. */

/*  LWORK   (local or global input) INTEGER */
/*          The dimension of the array WORK. */
/*          LWORK is local input and must be at least */
/*          LWORK >= NB_A * ( Mp0 + Nq0 + NB_A ), where */

/*          IROFF = MOD( IA-1, MB_A ), ICOFF = MOD( JA-1, NB_A ), */
/*          IAROW = INDXG2P( IA, MB_A, MYROW, RSRC_A, NPROW ), */
/*          IACOL = INDXG2P( JA, NB_A, MYCOL, CSRC_A, NPCOL ), */
/*          Mp0   = NUMROC( M+IROFF, MB_A, MYROW, IAROW, NPROW ), */
/*          Nq0   = NUMROC( N+ICOFF, NB_A, MYCOL, IACOL, NPCOL ), */

/*          and NUMROC, INDXG2P are ScaLAPACK tool functions; */
/*          MYROW, MYCOL, NPROW and NPCOL can be determined by calling */
/*          the subroutine BLACS_GRIDINFO. */

/*          If LWORK = -1, then LWORK is global input and a workspace */
/*          query is assumed; the routine only calculates the minimum */
/*          and optimal size for all work arrays. Each of these */
/*          values is returned in the first entry of the corresponding */
/*          work array, and no error message is issued by PXERBLA. */

/*  INFO    (global output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  If the i-th argument is an array and the j-entry had */
/*                an illegal value, then INFO = -(i*100+j), if the i-th */
/*                argument is a scalar and had an illegal value, then */
/*                INFO = -i. */

/*  Further Details */
/*  =============== */

/*  The matrix Q is represented as a product of elementary reflectors */

/*     Q = H(ja) H(ja+1) . . . H(ja+k-1), where k = min(m,n). */

/*  Each H(i) has the form */

/*     H(j) = I - tau * v * v' */

/*  where tau is a real scalar, and v is a real vector with v(1:i-1) = 0 */
/*  and v(i) = 1; v(i+1:m) is stored on exit in A(ia+i:ia+m-1,ja+i-1), */
/*  and tau in TAU(ja+i-1). */

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
    --work;
    --tau;
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
	    icoff = (*ja - 1) % desca[6];
	    iarow = indxg2p_(ia, &desca[5], &myrow, &desca[7], &nprow);
	    iacol = indxg2p_(ja, &desca[6], &mycol, &desca[8], &npcol);
	    i__1 = *m + (*ia - 1) % desca[5];
	    mp0 = numroc_(&i__1, &desca[5], &myrow, &iarow, &nprow);
	    i__1 = *n + icoff;
	    nq0 = numroc_(&i__1, &desca[6], &mycol, &iacol, &npcol);
	    lwmin = desca[6] * (mp0 + nq0 + desca[6]);

	    work[1] = (doublereal) lwmin;
	    lquery = *lwork == -1;
	    if (*lwork < lwmin && ! lquery) {
		*info = -9;
	    }
	}
	if (*lwork == -1) {
	    idum1[0] = -1;
	} else {
	    idum1[0] = 1;
	}
	idum2[0] = 9;
	pchk1mat_(m, &c__1, n, &c__2, ia, ja, &desca[1], &c__6, &c__1, idum1, 
		idum2, info);
    }

    if (*info != 0) {
	i__1 = -(*info);
	pxerbla_(&ictxt, "PDGEQRF", &i__1, (ftnlen)7);
	return 0;
    } else if (lquery) {
	return 0;
    }


/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return 0;
    }

    k = min(*m,*n);
    ipw = desca[6] * desca[6] + 1;
    pb_topget__(&ictxt, "Broadcast", "Rowwise", rowbtop, (ftnlen)9, (ftnlen)7,
	     (ftnlen)1);
    pb_topget__(&ictxt, "Broadcast", "Columnwise", colbtop, (ftnlen)9, (
	    ftnlen)10, (ftnlen)1);
    pb_topset__(&ictxt, "Broadcast", "Rowwise", "I-ring", (ftnlen)9, (ftnlen)
	    7, (ftnlen)6);
    pb_topset__(&ictxt, "Broadcast", "Columnwise", " ", (ftnlen)9, (ftnlen)10,
	     (ftnlen)1);

	// allocate memory on GPU for V and A2 and W
	double *V=NULL, *A2=NULL, *W=NULL;
	int ic, jc, iic, jjc, icrow, iccol; 
	int lda = desca[9];
	ic = 1;
	jc = 1 + desca[5];
	infog2l_(&ic, &jc, &desca[1], &nprow, &npcol, &myrow, &mycol, 
									&iic, &jjc, &icrow, &iccol);
	iic--;	jjc--;
	
	int descA2[9];
	int n_A2 = *n - desca[5];
	int izero = 0, ione = 1;
	descinit_ (descA2, m, &n_A2, &desca[5], &desca[5], &izero, &ione, &ictxt, &lda, &iinfo);
	if (iinfo!=0)
	{
		printf ("error at descinit_, %d, %s\n", __LINE__, __FILE__);
		exit(0);
	}
	i__1 = *m;
	int mpc = numroc_(&i__1, &descA2[4], &myrow, &izero, &nprow);
	i__1 = *n - descA2[4];
	int nqc = numroc_(&i__1, &descA2[5], &mycol, &ione, &npcol);
	
	double *pinnbuf=NULL;
	if (mpc*nqc>0)
	{
		//printf ("\n(%d,%d) mpc=%d, nqc=%d, ldc=%d\n", myrow, mycol, mpc, nqc, lda);
		TESTING_DEVALLOC (A2, double, mpc*nqc);
		TESTING_DEVALLOC (W, double, nqc*descA2[5]);
		TESTING_DEVALLOC (V, double, mpc*descA2[5]);
		TESTING_HOSTALLOC(pinnbuf, double, mpc*descA2[5]);

		//cublasSetMatrix(mpc, nqc, sizeof(double), &a[jjc*lda+iic+1], lda, A2, lda);
		
		cudaMemcpyAsync	(A2, &a[jjc*lda+iic+1],
				mpc*nqc*sizeof(double), cudaMemcpyHostToDevice, fstream);
	}

/*     Handle the first block of columns separately */

/* Computing MIN */
    i__1 = iceil_(ja, &desca[6]) * desca[6], i__2 = *ja + k - 1;
    jn = min(i__1,i__2);
    jb = jn - *ja + 1;

/*     Compute the QR factorization of the first block A(ia:ia+m-1,ja:jn) */
    pdgeqr2_(m, &jb, &a[1], ia, ja, &desca[1], &tau[1], &work[1], lwork, &
	    iinfo);

    if (*ja + jb <= *ja + *n - 1) {

/*        Form the triangular factor of the block reflector */
/*        H = H(ja) H(ja+1) . . . H(jn) */

	pdlarft_("Forward", "Columnwise", m, &jb, &a[1], ia, ja, &desca[1], &
		tau[1], &work[1], &work[ipw], (ftnlen)7, (ftnlen)10);

/*        Apply H' to A(ia:ia+m-1,ja+jb:ja+n-1) from the left */

	cudaStreamSynchronize	(fstream); 	
	i__1 = *n - jb;
	i__2 = *ja + jb;
	gpu_pdlarfb_("Left", "Transpose", "Forward", "Columnwise", m, &i__1, &jb, 
		&a[1], ia, ja, &desca[1], &work[1], &a[1], ia, &i__2, &desca[
		1], &work[ipw], (ftnlen)4, (ftnlen)9, (ftnlen)7, (ftnlen)10,
		A2, descA2, W, V, pinnbuf);
    }

/*     Loop over the remaining blocks of columns */

    i__1 = *ja + k - 1;
    i__2 = desca[6];
    for (j = jn + 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) 
	{
		/* Computing MIN */
		i__3 = k - j + *ja;
		jb = min(i__3,desca[6]);
		i__ = *ia + j - *ja;	// ia == ja == 1

		/*        Compute the QR factorization of the current block */
		/*        A(i:ia+m-1,j:j+jb-1) */

		i__3 = *m - j + *ja;
		pdgeqr2_(&i__3, &jb, &a[1], &i__, &j, &desca[1], &tau[1], &work[1], 
				lwork, &iinfo);

		if (j + jb <= *ja + *n - 1) 
		{

			/*           Form the triangular factor of the block reflector */
			/*           H = H(j) H(j+1) . . . H(j+jb-1) */

			i__3 = *m - j + *ja;
			pdlarft_("Forward", "Columnwise", &i__3, &jb, &a[1], &i__, &j, &
					desca[1], &tau[1], &work[1], &work[ipw], (ftnlen)7, (ftnlen)10);

			/*           Apply H' to A(i:ia+m-1,j+jb:ja+n-1) from the left */

			i__3 = *m - j + *ja;
			i__4 = *n - j - jb + *ja;
			i__5 = j + jb;
			
			gpu_pdlarfb_("Left", "Transpose", "Forward", "Columnwise", &i__3, &
					i__4, &jb, &a[1], &i__, &j, &desca[1], &work[1], &a[1], 
					&i__, &i__5, &desca[1], &work[ipw], 
					(ftnlen)4, (ftnlen)9, (ftnlen)7, (ftnlen)10,
					A2, descA2, W, V, pinnbuf);
					// i__ = j, i__5 = j+jb
		}

		/* L10: */
	}

    pb_topset__(&ictxt, "Broadcast", "Rowwise", rowbtop, (ftnlen)9, (ftnlen)7,
	     (ftnlen)1);
    pb_topset__(&ictxt, "Broadcast", "Columnwise", colbtop, (ftnlen)9, (
	    ftnlen)10, (ftnlen)1);

    work[1] = (doublereal) lwmin;

	if (mpc*nqc>0)
	{
		TESTING_HOSTFREE(pinnbuf);
		TESTING_DEVFREE(W);
		TESTING_DEVFREE(V);
		TESTING_DEVFREE(A2);
	}

	cudaStreamDestroy(fstream);

    return 0;

/*     End of PDGEQRF */

} /* origpdgeqrf_ */

