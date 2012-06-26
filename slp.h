#define DTYPE_	0
#define CTXT_	1
#define M_		2
#define N_		3
#define MB_		4
#define NB_		5
#define RSRC_	6
#define CSRC_	7
#define LLD_	8
#define TAG_L	1021
#define TAG_LL	904	

/********************************************************************/
extern void printmatrix_ (double * A, int * descA, char * NAME, int * len);
extern double get_cur_time();
extern void pdpotrrv_ (char * uplo, int * m, double * A, int * i, int * j, int * descA, double *work);

extern int iceil_(int * A, int * B);

//extern void pchk1mat_ (int * MA, int * MAPOSO, int * NA, int * NAPOS0, int * IA, int * JA, int * DESCA, 
//						int * DESCAPOS0, int * NEXTRA, int * EX, int * EXPOS, int * INFO);

//extern void chk1mat_ (int * MA, int * MAPOSO, int * NA, int * NAPOS0, int * IA, int * JA, int * DESCA, int * DESCAPOS0, int * INFO);

extern void Cblacs_gridexit( int ictxt );

extern void Cblacs_pinfo( int * iam, int * nprocs );

extern void Cblacs_get( int, int, int * ictxt );

extern void Cblacs_gridinit( int *, char *, int nprow, int npcol );

extern void Cblacs_gridinfo( int ictxt, int * nprow, int * npcol, int * myrow, int * mycol );

extern void descinit_ (int * desc, int * M, int * N, int * MB, int * NB, int * IRSRC, int * ICSRC, int * ICTXT, int * LLD, int * INFO);

extern void infog2l_ (int *GRINDX, int *GCINDX, int *DESC, int *NPROW, int *NPCOL, int *MYROW, int *MYCOL, 
					   int *LRINDX, int *LCINDX, int *RSRC, int *CSRC );

extern void infog1l_ (int *GRINDX, int * NB, int * NPROC, int * MYPROC, int *ISRCPROC, int *LINDX, int *ROCSRC); 

extern void Cblacs_gridinfo(int ConTxt, int *nprow, int *npcol, int *myrow, int *mycol);

extern void pdlaprnt_ (int *m, int * n, double * A, int * ia, int * ja, int * descA, int * prow, int * pcol, 
						char * NAME, int * NOUT, double * work, int len);
extern void pdmatadd_ ( int *, int *, double *, double *, int *, int *, int *, double *, double *, int *, int *, int *);

extern void Cdgebs2d(int ConTxt, char *scope, char *top, int m, int n, double *A, int lda);

extern void	pdmatgen_ (int *, char *, char *, int *, int *, int *, int *, double *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
extern void Cdgebr2d(int ConTxt, char *scope, char *top, int m, int n, double *A, int lda, int rsrc, int csrc);

extern void Cdtrbs2d(int ConTxt, char *scope, char *top, char *uplo, char *diag, int m, int n, double *A, int lda);

extern void Cdtrbr2d(int ConTxt, char *scope, char *top, char *uplo, char *diag, int m, int n, double *A, int lda, int rsrc, int csrc);

//extern void pb_topget__( int * ICTXT, char * OP, char * SCOPE, char * TOP );

//extern void pb_topset__( int * ICTXT, char * OP, char * SCOPE, char * TOP );

extern int numroc_(int * N, int * NB, int * IPROC, int * ISRCPROC, int * NPROCS );
		
extern void dgemm(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc);

extern void dtrsm_ (const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const double *alpha, const double *a, const int *lda, double *b, const int *ldb);

extern void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *beta, double *c, const int *ldc);

extern void dscal_ (int *, double *, double *, int *);

extern void pdgemm_( char * TRANSA, char * TRANSB,	int * M, int * N, int * K,double * ALPHA, double * A, int * IA, int * JA, int * DESCA,
					double * B, int * IB, int * JB, int * DESCB, double * BETA,	double * C, int * IC, int * JC, int * DESCC );

extern void igamn2d_ (int * ICTXT, char * SCOPE, char * TOP, int * M, int * N, int * A, int * LDA, int * RA, int * CA, 
						int * RCFLAG, int * RDEST, int * CDEST);

extern int indxl2g_( int * INDXLOC, int * NB, int * IPROC, int * ISRCPROC, int * NPROCS );

extern double	pdlange_(char *norm, int *m, int *n, double *a, int *ia, int *ja, int *desca, double *work);

double dlange_ (char *norm, int *m, int *n, double *A, int *lda, double *work);

extern void Cblacs_set(int ConTxt, int what, int *val);

extern void pdlacpy_ (char *, int *, int *, double *, int *, int *, int *, double *, int *, int *, int *);

extern void pdelset_ (double *A, int *IA, int *JA, int *descA, double *alpha);
extern void pdelget_ (char *scope, char *top, double *alpha, double *A, int *IA, int *JA, int *descA);
extern void pddot_( int * N, double * DOT, double * X, int * IX, int * JX, int * DESCX, int * INCX, double * Y, int * IY, int * JY, int * DESCY, int * INCY );

extern void pdcopy_( int * N, double * X, int * IX, int * JX, int * DESCX, int * INCX,double * Y, int * IY, int * JY, int * DESCY, int * INCY );


extern void Cpdgemr2d(int m, int n, double *A, int IA, int JA, int *descA, double *B, int IB, int JB, int *descB, int gcontext);


extern void pdgeadd_( char *TRANS, int * M, int * N,  double * ALPHA, double * A, int * IA, int * JA, int * DESCA, 
												double * BETA, double * C, int * IC, int * JC, int * DESCC ); //sub( C ) := beta*sub( C ) + alpha*op( sub( A ) )

extern void pdgesv_ (int *n, int *nrhs, double *A, int *ia, int *ja, int *descA, int *ipiv, double *B, int *ib, int *jb, int *descB, int *info);

extern void pdlaswp_ (char *direc, char *rwocol, int *n, double *A, int *ia, int *ja, int *descA, int *k1, int *k2, int *ipiv);

extern void pdgetrf_ (int *M, int *N, double *A, int *ia, int *ja, int *descA, int *ipiv, int *info);

extern void origpdgetrf_ (int *M, int *N, double *A, int *ia, int *ja, int *descA, int *ipiv, int *info);

extern void gpu_pdgetrf_ (int *M, int *N, double *A, int *ia, int *ja, int *descA, int *ipiv, int *info);

extern void pdtrsm_( char * SIDE, char * UPLO, char * TRANS, char * DIAG, int * M, int * N, double * ALPHA, double * A, int * IA, int * JA, int * DESCA,
									              double * B, int * IB, int * JB, int * DESCB );

extern void pdgeqrf_ (int *m, int *n, double *A, int *ia, int *ja, int *descA, double *tau, double *work, int *lwork, int *info);

extern void pdormqr_ (char *side, char *tran, int *m, int *n, int *k, double *A, int *ia, int *ja, int *descA,
						double *tau, double *C, int *ic, int *jc, int *descC, double *work, int *lwork, int *info);

extern void pdtrmm_ (char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *ia, int *ja, int *desca, double *B, int *ib, int *jb, int *descb );

extern void pdamax_( int * N, double * AMAX, int * INDX, double * X, int * IX, int * JX, int * DESCX, int * INCX );

extern void pdlapiv_ (char * direc, char * rowcol, char * pivroc, int *m, int *n, double *A, int *ia, int *ja, int *descA, int *ipiv, int *ip,int *jp, int *descIP, int *iwork);
