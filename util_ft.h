/*	UTILITY ROUTINE INCLUDE for SCALAPACK and PLAPACK testing	*/

#include "mpi.h"
#include <stdlib.h>
#include "stdio.h" 
#include <sys/time.h>
#include "string.h"
#include "setjmp.h"
#include "stdbool.h" // really?

#define THRESHOLD 1e-6

#define ISWORKER (myrank_mpi<=2 || (myrank_mpi>=4 && myrank_mpi<=6))
#define MIN(a,b) (a>b)?b:a
#define MAX(a,b) (a>b)?a:b

#define A_error()	A[(lcindx-1)*np_A+lrindx-1] = -99; \
						printf ("setting A(%d, %d) to -99 on proc (%d, %d), local(%d, %d)\n", gdeadrow, gdeadcol, rsrc, csrc, lrindx, lcindx);
						
#define B_error()	B[(lcindx-1)*np_B+lrindx-1] = -99; \
						printf ("setting B(%d, %d) to -99 on proc (%d, %d), local(%d, %d)\n", gdeadrow, gdeadcol, rsrc, csrc, lrindx, lcindx);

#define ABORT  {printf ("aborting on line %d in %s\n", __LINE__, __FILE__);\
			MPI_Abort (MPI_COMM_WORLD, -1);}

#define ROOTSAY(k) if (grid->myrank_mpi==0) printf ("%s\n", k);


#define CHECKSUM 1
#define NCHECKSUM 0

#define ZONE1 1
#define NZONE1 0

#define ZONE2 1
#define NZONE2 0

#define ZONE3 1
#define NZONE3 0

#define COPY 1
#define NOCOPY 0

//#define blacs_gridinfo__ blacs_gridinfo_
//#define pb_topget__ pb_topget_
//#define pb_topset__ pb_topset_

typedef struct s_FTKit
{
	double *A;
	double *localcopy;
	double *tau;
	double *work;
	int lwork;
}t_FTKit;

typedef struct s_Grid
{
	int ictxt;
	int nprocs_mpi;
	int myrank_mpi;
	int myrow;
	int mycol;
	int nprow;
	int npcol;
	int nb;
}t_Grid;

typedef struct s_Comm
{
	int created;
	MPI_Comm part_comm;
	int part_root;
}t_Comm;

typedef struct 
{
	int Nc;
	int Mc;
	int IsMskew;
	int IsNskew;
	int MaxLocalCol;
	int MaxLocalRow;
	t_Comm * vcomm;
	t_Comm * hcomm;
	MPI_Comm rowcomm;
	MPI_Comm colcomm;
	MPI_Group colgroup;
	MPI_Group rowgroup;
	double *S;
	double *R;
	double *localcopy;
	int descL[9];
	int np_L, nq_L;
	t_FTKit kit;
}t_checksuite;

extern void checkpointing (char side,
					double *A, int MA, int NA, int IA, int JA, int *descA, 
					double *Acd, int IAcd, int JAcd, int *descAcd,
					t_checksuite *cs, t_Grid *grid);

extern void checkpointing_short (char side,
					double *A, int MA, int NA, int IA, int JA, int *descA, 
					double *Acd, int IAcd, int JAcd, int *descAcd,
					t_checksuite *cs, t_Grid *grid, int errstep);

extern void set_grid (t_Grid *grid, int ictxt, int nprocs_mpi, int myrank_mpi, int myrow, int mycol, int nprow, int npcol, int nb);


extern void generator_check (char side, char type, double **G, int *descG, 
								 double **H, int *descH, 
								 int M, int N, t_Grid *grid,
								 int * np_G, int *nq_G,
								 int * np_H, int *nq_H); 

extern void distr_matrix (bool fill,	double **A, int *descA, 
								int M, int N, t_Grid *grid,
								int *np_A, int *nq_A); 

extern void check_error (char side,
				  double *A, int MA, int NA, int iA, int jA, int *descA,
				  t_checksuite *cs, int noe,
				  t_Grid *grid, int *error_where	// 0: No, 1: L, 2:U
				  );

extern int get_recovery_node (int *err_cords, int noe);


/*
extern void recover (bool relative, double *C, int MC, int NC, int iC, int jC, int *descC, 
			  double *Gr, int MGr, int NGr, int iGr, int jGr, int *descGr, 
			int *err_x, int *err_y, t_Grid *grid, int noedx, int noedy, int noe);
			*/

extern void recover_zone1 (double *A, int m, int n, int *descA, int errx, int erry, int errstep, int vic_id, t_Grid *grid, t_checksuite *cs);

extern void recover_zone2 (double *A, int m, int n, int *descA, int errx, int erry, int errstep, int vic_id, t_Grid *grid, t_checksuite *cs);

extern void recover_zone2_tail (double *A, int m, int n, int *descA, int errx, int erry, int errstep, int vic_id, t_Grid *grid, t_checksuite *cs);

extern void recover_zone3 (double *A, int m, int n, int *descA, int errx, int erry, int errstep, int vic_id, t_Grid *grid, t_checksuite *cs);

extern void recover_checksum (char side, double *A, int m, int n, int *descA, int errx, int erry, int errstep, int vic_id, t_Grid *grid, t_checksuite *cs);

extern void recover (int CHK, int Z1, int Z2, int Z3, double *A, int m, int n, int *descA, int errx, int erry, int errstep, t_Grid *grid, t_checksuite *cs);

extern double verifyLU(double *Aorg, double * A, int M, int N, int * descA, int * ipiv, t_Grid *grid);
extern double verifyQR(double *Aorg, double * A, int M, int N, int * descA, double * tau, t_Grid *grid);

extern void pdgetrrv_ (int *m, int *n, double *A, int *ia, int *ja, int *descA, int *ipiv, double *work);
extern void pdgeqrrv_ (int *m, int *n, double *A, int *ia, int *ja, int *descA, double*tau, double *work);

extern int indxg2p_ (int *,int *, int *, int *, int *);

extern void pdlafchk_ (char *aform, char *diag, int *M, int *N, double *A, int *ia, int *ja, int *descA, int *iaseed, double *anorm, double *fresid, double *work);

extern void get_checksuite (int Nc, int Mc, int IsNskew, int IsMskew, int MaxLocalCol, int MaxLocalRow, t_checksuite *cs, t_Grid *grid);

extern void del_checksuite (t_checksuite *cs, int nb);

extern void inject_errors (double *C, int *descC, int *err_XX, int *err_YY, t_Grid *grid);

extern void fprintmatrix_ (double * A, int m, int n, int * descA, char * NAME, int len);
extern void kill_proc(double *A, int *descA, t_Grid *grid);
extern void verify_checkpointing (int M, int N, double *A, int *descA, t_Grid *grid, t_checksuite *cs);
extern void local_copy(double *A, int *descA, int IA, int JA, t_checksuite *cs, t_Grid *grid);
extern void panel_reset(double *A, int *descA, int IA, int JA, t_checksuite *cs, t_Grid *grid);
extern void dump_to_disk (double *A, int *descA, double *tau, double *work, int lwork, t_Grid *grid, t_checksuite *cs);
extern void load_from_disk(double *A, int *descA, double *tau, double *work, int lwork, t_Grid *grid, t_checksuite *cs);
extern void hor_checkpointing_local (double *A, int *descA, int IA, int JA, int iscopy, t_checksuite *cs, t_Grid *grid);
extern void recover_local (char side, double *A, int MA, int NA, int *descA, int errx, int erry, int errstep, int vic_id, t_Grid *grid, t_checksuite *cs);
extern void recover_local_fillin (double *A, int MA, int NA, int *descA, int errx, int erry, int errstep, int vic_id, t_Grid *grid, t_checksuite *cs);
extern void set_FTKit (double *A, t_checksuite *cs, double *tau, double *work, int lwork);
