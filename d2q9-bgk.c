#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
// #define DEBUG true


typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

// typedef struct
// {
//   float speeds[NSPEEDS];
// } t_speed;

typedef struct
{
  float* speeds0;
  float* speeds1;
  float* speeds2;
  float* speeds3;
  float* speeds4;
  float* speeds5;
  float* speeds6;
  float* speeds7;
  float* speeds8;
} t_speed;


int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
float fusion(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);
int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);
float total_density(const t_param params, t_speed* cells);
float av_velocity(const t_param params, t_speed* cells, int* obstacles);
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);


int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed cells;    /* grid containing fluid densities */
  t_speed tmp_cells;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  //omp_set_num_threads(16);

/* parse the command line */
if (argc != 3)
{
  usage(argv[0]);
}
else
{
  paramfile = argv[1];
  obstaclefile = argv[2];
}

/* initialise our data structures and load values from file */
initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

/* iterate for maxIters timesteps */
gettimeofday(&timstr, NULL);
tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

for (int tt = 0; tt < params.maxIters; tt=tt+2)
{
//accelerate_flow(params, cells, obstacles);
av_vels[tt] = fusion(params, &cells, &tmp_cells, obstacles);


//accelerate_flow(params, tmp_cells, obstacles);
av_vels[tt+1] = fusion(params, &tmp_cells, &cells, obstacles);

#ifdef DEBUG
  printf("==timestep: %d==\n", tt);
  printf("av velocity: %.12E\n", av_vels[tt]);
  printf("tot density: %.12E\n", total_density(params, &cells));
#endif
}

gettimeofday(&timstr, NULL);
toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
getrusage(RUSAGE_SELF, &ru);
timstr = ru.ru_utime;
usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
timstr = ru.ru_stime;
systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

/* write final values and free memory */
printf("==done==\n");
printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, &cells, obstacles));
printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
write_values(params, &cells, obstacles, av_vels);
finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

return EXIT_SUCCESS;
}



// int accelerate_flow(const t_param params, t_speed*  cells, int*  obstacles)
// {
//   /* compute weighting factors */
//   float w1 = params.density * params.accel / 9.f;
//   float w2 = params.density * params.accel / 36.f;
//
//   /* modify the 2nd row of the grid */
//   int jj = params.ny - 2;
//
//   for (int ii = 0; ii < params.nx; ii++)
//   {
//     /* if the cell is not occupied and
//     ** we don't send a negative density */
//     if (!obstacles[ii + jj*params.nx]
//         && (cells->speeds3[ii + jj*params.nx] - w1) > 0.f
//         && (cells->speeds6[ii + jj*params.nx] - w2) > 0.f
//         && (cells->speeds7[ii + jj*params.nx] - w2) > 0.f)
//     {
//       /* increase 'east-side' densities */
//       cells->speeds1[ii + jj*params.nx] += w1;
//       cells->speeds5[ii + jj*params.nx] += w2;
//       cells->speeds8[ii + jj*params.nx] += w2;
//       /* decrease 'west-side' densities */
//       cells->speeds3[ii + jj*params.nx] -= w1;
//       cells->speeds6[ii + jj*params.nx] -= w2;
//       cells->speeds7[ii + jj*params.nx] -= w2;
//     }
//   }
//
//   return EXIT_SUCCESS;
// }


float fusion(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles){

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  const float w4 = 2.f * c_sq;
  const float w3 =w4 * c_sq;

  float a1 = params.density * params.accel / 9.f;
  float a2 = params.density * params.accel / 36.f;
  int jj = params.ny - 2;

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u =0.f;         /* accumulated magnitudes of velocity for each cell */

  __assume_aligned((cells->speeds0), 64);
  __assume_aligned(cells->speeds1, 64);
  __assume_aligned(cells->speeds2, 64);
  __assume_aligned(cells->speeds3, 64);
  __assume_aligned(cells->speeds4, 64);
  __assume_aligned(cells->speeds5, 64);
  __assume_aligned(cells->speeds6, 64);
  __assume_aligned(cells->speeds7, 64);
  __assume_aligned(cells->speeds8, 64);

  __assume_aligned(tmp_cells->speeds0, 64);
  __assume_aligned(tmp_cells->speeds1, 64);
  __assume_aligned(tmp_cells->speeds2, 64);
  __assume_aligned(tmp_cells->speeds3, 64);
  __assume_aligned(tmp_cells->speeds4, 64);
  __assume_aligned(tmp_cells->speeds5, 64);
  __assume_aligned(tmp_cells->speeds6, 64);
  __assume_aligned(tmp_cells->speeds7, 64);
  __assume_aligned(tmp_cells->speeds8, 64);

  __assume(params.nx%2==0);
  __assume(params.nx%4==0);
  __assume(params.nx%16==0);



  #pragma omp parallel num_threads(28) reduction(+:tot_u,tot_cells)
  {

  #pragma omp for nowait schedule(static)
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speeds3[ii + jj*params.nx] - a1) > 0.f
        && (cells->speeds6[ii + jj*params.nx] - a2) > 0.f
        && (cells->speeds7[ii + jj*params.nx] - a2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speeds1[ii + jj*params.nx] += a1;
      cells->speeds5[ii + jj*params.nx] += a2;
      cells->speeds8[ii + jj*params.nx] += a2;
      /* decrease 'west-side' densities */
      cells->speeds3[ii + jj*params.nx] -= a1;
      cells->speeds6[ii + jj*params.nx] -= a2;
      cells->speeds7[ii + jj*params.nx] -= a2;
    }
  }



  #pragma omp for nowait schedule(static)
  // #pragma simd aligned
  for (int jj = 0; jj < params.ny; jj++)
  {


    __assume_aligned((cells->speeds0), 64);
    __assume_aligned(cells->speeds1, 64);
    __assume_aligned(cells->speeds2, 64);
    __assume_aligned(cells->speeds3, 64);
    __assume_aligned(cells->speeds4, 64);
    __assume_aligned(cells->speeds5, 64);
    __assume_aligned(cells->speeds6, 64);
    __assume_aligned(cells->speeds7, 64);
    __assume_aligned(cells->speeds8, 64);

    __assume_aligned(tmp_cells->speeds0, 64);
    __assume_aligned(tmp_cells->speeds1, 64);
    __assume_aligned(tmp_cells->speeds2, 64);
    __assume_aligned(tmp_cells->speeds3, 64);
    __assume_aligned(tmp_cells->speeds4, 64);
    __assume_aligned(tmp_cells->speeds5, 64);
    __assume_aligned(tmp_cells->speeds6, 64);
    __assume_aligned(tmp_cells->speeds7, 64);
    __assume_aligned(tmp_cells->speeds8, 64);

    __assume(params.nx%2==0);
    __assume(params.nx%4==0);
    __assume(params.nx%16==0);


    //#pragma simd
  //  #pragma vector aligned
    #pragma omp simd reduction(+:tot_u,tot_cells)
    for (int ii = 0; ii < params.nx; ii++)
    {




      //PROPAGATE VARIABLES
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      //printf("cell check: %d==\n", cells->speeds3[x_e + jj*params.nx]);



      //////////////////////////////////
      //////////REBOUND////////////////
      /////////////////////////////////
      // if (obstacles[jj*params.nx + ii])
      // {
        // tmp_cells->speeds1[ii + jj*params.nx] = cells->speeds3[x_e + jj*params.nx];
        // tmp_cells->speeds2[ii + jj*params.nx] = cells->speeds4[ii + y_n*params.nx];
        // tmp_cells->speeds3[ii + jj*params.nx] = cells->speeds1[x_w + jj*params.nx];
        // tmp_cells->speeds4[ii + jj*params.nx] = cells->speeds2[ii + y_s*params.nx];
        // tmp_cells->speeds5[ii + jj*params.nx] = cells->speeds7[x_e + y_n*params.nx];
        // tmp_cells->speeds6[ii + jj*params.nx] = cells->speeds8[x_w + y_n*params.nx];
        // tmp_cells->speeds7[ii + jj*params.nx] = cells->speeds5[x_w + y_s*params.nx];
        // tmp_cells->speeds8[ii + jj*params.nx] = cells->speeds6[x_e + y_s*params.nx];
      // }else
      // {

        /* compute local density total */
        float local_density = 0.f;

        local_density = local_density + cells->speeds0[ii + jj*params.nx] + cells->speeds1[x_w + jj*params.nx] + cells->speeds2[ii + y_s*params.nx] + cells->speeds3[x_e + jj*params.nx] + cells->speeds4[ii + y_n*params.nx] + cells->speeds5[x_w + y_s*params.nx] + cells->speeds6[x_e + y_s*params.nx] + cells->speeds7[x_e + y_n*params.nx] + cells->speeds8[x_w + y_n*params.nx];

        //printf("density check: %d==\n", local_density);

        /* compute x velocity component */
        float u_x = (cells->speeds1[x_w + jj*params.nx]
                      + cells->speeds5[x_w + y_s*params.nx]
                      + cells->speeds8[x_w + y_n*params.nx]
                      - (cells->speeds3[x_e + jj*params.nx]
                         + cells->speeds6[x_e + y_s*params.nx]
                         + cells->speeds7[x_e + y_n*params.nx]))
                     / local_density;

        /* compute y velocity component */
        float u_y = (cells->speeds2[ii + y_s*params.nx]
                      + cells->speeds5[x_w + y_s*params.nx]
                      + cells->speeds6[x_e + y_s*params.nx]
                      - (cells->speeds4[ii + y_n*params.nx]
                         + cells->speeds7[x_e + y_n*params.nx]
                         +cells->speeds8[x_w + y_n*params.nx]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */

        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / w4);

        // for (size_t kk = 1; kk < 5; kk++) {
        //   d_equ[kk] = w1 * local_density * (1.f + u[kk] / c_sq + (u[kk]*u[kk]) / w3 - u_sq / w4);
        // }
        //
        // for (size_t kk = 5; kk < 9; kk++) {
        //   d_equ[kk] = w2 * local_density * (1.f + u[kk] / c_sq + (u[kk]*u[kk]) / w3 - u_sq / w4);
        // }

        #pragma novector
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq + (u[1]*u[1]) / w3 - u_sq / w4);
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq + (u[2]*u[2]) / w3 - u_sq / w4);
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq + (u[3]*u[3]) / w3 - u_sq / w4);
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq + (u[4]*u[4]) / w3 - u_sq / w4);

        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq + (u[5]*u[5]) / w3 - u_sq / w4);
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq + (u[6]*u[6]) / w3 - u_sq / w4);
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq + (u[7]*u[7]) / w3 - u_sq / w4);
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq + (u[8]*u[8]) / w3 - u_sq / w4);



        // tmp_cells->speeds0[ii + jj*params.nx] = cells->speeds0[ii + jj*params.nx] + params.omega * (d_equ[0] - cells->speeds0[ii + jj*params.nx]);
        // tmp_cells->speeds1[ii + jj*params.nx] = cells->speeds1[x_w + jj*params.nx]  + params.omega * (d_equ[1] - cells->speeds1[x_w + jj*params.nx] );
        // tmp_cells->speeds2[ii + jj*params.nx] = cells->speeds2[ii + y_s*params.nx]  + params.omega * (d_equ[2] - cells->speeds2[ii + y_s*params.nx] );
        // tmp_cells->speeds3[ii + jj*params.nx] = cells->speeds3[x_e + jj*params.nx] + params.omega * (d_equ[3] -cells->speeds3[x_e + jj*params.nx] );
        // tmp_cells->speeds4[ii + jj*params.nx] = cells->speeds4[ii + y_n*params.nx] + params.omega * (d_equ[4] -cells->speeds4[ii + y_n*params.nx] );
        // tmp_cells->speeds5[ii + jj*params.nx] = cells->speeds5[x_w + y_s*params.nx] + params.omega * (d_equ[5] -cells->speeds5[x_w + y_s*params.nx] );
        // tmp_cells->speeds6[ii + jj*params.nx] = cells->speeds6[x_e + y_s*params.nx] + params.omega * (d_equ[6] -cells->speeds6[x_e + y_s*params.nx] );
        // tmp_cells->speeds7[ii + jj*params.nx] = cells->speeds7[x_e + y_n*params.nx] + params.omega * (d_equ[7] - cells->speeds7[x_e + y_n*params.nx]);
        // tmp_cells->speeds8[ii + jj*params.nx] = cells->speeds8[x_w + y_n*params.nx] + params.omega * (d_equ[8] -cells->speeds8[x_w + y_n*params.nx] );
        //
        //
        //
        // tmp_cells->speeds2[ii + jj*params.nx] =  cells->speeds4[ii + y_n*params.nx];
        // tmp_cells->speeds3[ii + jj*params.nx] = cells->speeds1[x_w + jj*params.nx];
        // tmp_cells->speeds4[ii + jj*params.nx] = cells->speeds2[ii + y_s*params.nx];
        // tmp_cells->speeds5[ii + jj*params.nx] = cells->speeds7[x_e + y_n*params.nx];
        // tmp_cells->speeds6[ii + jj*params.nx] = cells->speeds8[x_w + y_n*params.nx];
        // tmp_cells->speeds7[ii + jj*params.nx] = cells->speeds5[x_w + y_s*params.nx];
        // tmp_cells->speeds8[ii + jj*params.nx] = cells->speeds6[x_e + y_s*params.nx];

        tmp_cells->speeds0[ii + jj*params.nx] = (!obstacles[jj*params.nx + ii]) ? cells->speeds0[ii + jj*params.nx] + params.omega * (d_equ[0] - cells->speeds0[ii + jj*params.nx]) :   tmp_cells->speeds0[ii + jj*params.nx];
        tmp_cells->speeds1[ii + jj*params.nx] = (obstacles[jj*params.nx + ii]) ? cells->speeds3[x_e + jj*params.nx] : cells->speeds1[x_w + jj*params.nx]  + params.omega * (d_equ[1] - cells->speeds1[x_w + jj*params.nx] );
        tmp_cells->speeds2[ii + jj*params.nx] = (obstacles[jj*params.nx + ii]) ? cells->speeds4[ii + y_n*params.nx] : cells->speeds2[ii + y_s*params.nx]  + params.omega * (d_equ[2] - cells->speeds2[ii + y_s*params.nx] );
        tmp_cells->speeds3[ii + jj*params.nx] = (obstacles[jj*params.nx + ii]) ? cells->speeds1[x_w + jj*params.nx] : cells->speeds3[x_e + jj*params.nx] + params.omega * (d_equ[3] -cells->speeds3[x_e + jj*params.nx] );
        tmp_cells->speeds4[ii + jj*params.nx] = (obstacles[jj*params.nx + ii]) ? cells->speeds2[ii + y_s*params.nx] : cells->speeds4[ii + y_n*params.nx] + params.omega * (d_equ[4] -cells->speeds4[ii + y_n*params.nx] );
        tmp_cells->speeds5[ii + jj*params.nx] = (obstacles[jj*params.nx + ii]) ? cells->speeds7[x_e + y_n*params.nx] : cells->speeds5[x_w + y_s*params.nx] + params.omega * (d_equ[5] -cells->speeds5[x_w + y_s*params.nx] );
        tmp_cells->speeds6[ii + jj*params.nx] = (obstacles[jj*params.nx + ii]) ? cells->speeds8[x_w + y_n*params.nx] : cells->speeds6[x_e + y_s*params.nx] + params.omega * (d_equ[6] -cells->speeds6[x_e + y_s*params.nx] );
        tmp_cells->speeds7[ii + jj*params.nx] = (obstacles[jj*params.nx + ii]) ? cells->speeds5[x_w + y_s*params.nx] : cells->speeds7[x_e + y_n*params.nx] + params.omega * (d_equ[7] - cells->speeds7[x_e + y_n*params.nx]);
        tmp_cells->speeds8[ii + jj*params.nx] = (obstacles[jj*params.nx + ii]) ? cells->speeds6[x_e + y_s*params.nx] : cells->speeds8[x_w + y_n*params.nx] + params.omega * (d_equ[8] -cells->speeds8[x_w + y_n*params.nx] );




      //  printf("%d\n", ((u_x * u_x) + (u_y * u_y)) );
      tot_u = (!obstacles[jj*params.nx + ii]) ? (tot_u + sqrtf((u_x * u_x) + (u_y * u_y))): tot_u;
      tot_cells = (!obstacles[jj*params.nx + ii]) ? tot_cells+1 : tot_cells;

      }
    }
  }
// }
//  printf("%f\n", (tot_u / (float)tot_cells) );

  //printf("%f\n", tot_u / (float)tot_cells);
  return tot_u / (float)tot_cells;

}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        local_density = local_density + cells->speeds0[ii + jj*params.nx] + cells->speeds1[ii + jj*params.nx] + cells->speeds2[ii + jj*params.nx] + cells->speeds3[ii + jj*params.nx] + cells->speeds4[ii + jj*params.nx] + cells->speeds5[ii + jj*params.nx] + cells->speeds6[ii + jj*params.nx] + cells->speeds7[ii + jj*params.nx] + cells->speeds8[ii + jj*params.nx];



                     /* compute x velocity component */
                     float u_x = (cells->speeds1[ii + jj*params.nx]
                                   + cells->speeds5[ii + jj*params.nx]
                                   + cells->speeds8[ii + jj*params.nx]
                                   - (cells->speeds3[ii + jj*params.nx]
                                      + cells->speeds6[ii + jj*params.nx]
                                      + cells->speeds7[ii + jj*params.nx]))
                                  / local_density;

                     /* compute y velocity component */
                     float u_y = (cells->speeds2[ii + jj*params.nx]
                                   + cells->speeds5[ii + jj*params.nx]
                                   + cells->speeds6[ii + jj*params.nx]
                                   - (cells->speeds4[ii + jj*params.nx]
                                      + cells->speeds7[ii + jj*params.nx]
                                      +cells->speeds8[ii + jj*params.nx]))
                                  / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}


int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells, t_speed* tmp_cells,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */

  // cells = malloc(sizeof(float*) * 9);
  cells->speeds0 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  cells->speeds1 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  cells->speeds2 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  cells->speeds3 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  cells->speeds4 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  cells->speeds5 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  cells->speeds6 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  cells->speeds7 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  cells->speeds8 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);

  if (cells == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
   // (*tmp_cells) = malloc(sizeof(float*) * 9);
  tmp_cells->speeds0 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  tmp_cells->speeds1 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  tmp_cells->speeds2 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  tmp_cells->speeds3 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  tmp_cells->speeds4 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  tmp_cells->speeds5 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  tmp_cells->speeds6 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  tmp_cells->speeds7 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);
  tmp_cells->speeds8 = (float*)_mm_malloc(sizeof(float) * (params->ny *params -> nx), 64);

  if (tmp_cells == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

 #pragma omp parallel num_threads(28)
 {

 #pragma omp for nowait schedule(static)
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      cells->speeds0[ii + jj*params->nx] = w0;
      /* axis directions */
      cells->speeds1[ii + jj*params->nx] = w1;
      cells->speeds2[ii + jj*params->nx] = w1;
      cells->speeds3[ii + jj*params->nx] = w1;
      cells->speeds4[ii + jj*params->nx] = w1;
      /* diagonals */
      cells->speeds5[ii + jj*params->nx] = w2;
      cells->speeds6[ii + jj*params->nx] = w2;
      cells->speeds7[ii + jj*params->nx] = w2;
      cells->speeds8[ii + jj*params->nx] = w2;
    }
  }

  #pragma omp for nowait schedule(static)
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }
}


  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed* cells, t_speed* tmp_cells,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */

  _mm_free (cells->speeds0);
  cells->speeds0 = NULL;
  _mm_free (cells->speeds1);
  cells->speeds1 = NULL;
  _mm_free (cells->speeds2);
  cells->speeds2 = NULL;
  _mm_free (cells->speeds3);
  cells->speeds3 = NULL;
  _mm_free (cells->speeds4);
  cells->speeds4 = NULL;
  _mm_free (cells->speeds5);
  cells->speeds5 = NULL;
  _mm_free (cells->speeds6);
  cells->speeds6 = NULL;
  _mm_free (cells->speeds7);
  cells->speeds7 = NULL;
  _mm_free (cells->speeds8);
  cells->speeds8 = NULL;

  _mm_free (tmp_cells->speeds0);
  tmp_cells->speeds0 = NULL;
  _mm_free (tmp_cells->speeds1);
  tmp_cells->speeds1 = NULL;
  _mm_free (tmp_cells->speeds2);
  tmp_cells->speeds2 = NULL;
  _mm_free (tmp_cells->speeds3);
  tmp_cells->speeds3 = NULL;
  _mm_free (tmp_cells->speeds4);
  tmp_cells->speeds4 = NULL;
  _mm_free (tmp_cells->speeds5);
  tmp_cells->speeds5 = NULL;
  _mm_free (tmp_cells->speeds6);
  tmp_cells->speeds6 = NULL;
  _mm_free (tmp_cells->speeds7);
  tmp_cells->speeds7 = NULL;
  _mm_free (tmp_cells->speeds8);
  tmp_cells->speeds8 = NULL;


  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
  total = total + cells->speeds0[ii + jj*params.nx] + cells->speeds1[ii + jj*params.nx] + cells->speeds2[ii + jj*params.nx] + cells->speeds3[ii + jj*params.nx] + cells->speeds4[ii + jj*params.nx] + cells->speeds5[ii + jj*params.nx] + cells->speeds6[ii + jj*params.nx] + cells->speeds7[ii + jj*params.nx] + cells->speeds8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;


  local_density = local_density + cells->speeds0[ii + jj*params.nx] + cells->speeds1[ii + jj*params.nx] + cells->speeds2[ii + jj*params.nx] + cells->speeds3[ii + jj*params.nx] + cells->speeds4[ii + jj*params.nx] + cells->speeds5[ii + jj*params.nx] + cells->speeds6[ii + jj*params.nx] + cells->speeds7[ii + jj*params.nx] + cells->speeds8[ii + jj*params.nx];


  /* compute x velocity component */
   u_x = (cells->speeds1[ii + jj*params.nx]
                + cells->speeds5[ii + jj*params.nx]
                + cells->speeds8[ii + jj*params.nx]
                - (cells->speeds3[ii + jj*params.nx]
                   + cells->speeds6[ii + jj*params.nx]
                   + cells->speeds7[ii + jj*params.nx]))
               / local_density;

  /* compute y velocity component */
   u_y = (cells->speeds2[ii + jj*params.nx]
                + cells->speeds5[ii + jj*params.nx]
                + cells->speeds6[ii + jj*params.nx]
                - (cells->speeds4[ii + jj*params.nx]
                   + cells->speeds7[ii + jj*params.nx]
                   +cells->speeds8[ii + jj*params.nx]))
               / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
