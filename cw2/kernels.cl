#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jj* nx]
      && (cells[ii + jj* nx].speeds[3] - w1) > 0.f
      && (cells[ii + jj* nx].speeds[6] - w2) > 0.f
      && (cells[ii + jj* nx].speeds[7] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[ii + jj* nx].speeds[1] += w1;
    cells[ii + jj* nx].speeds[5] += w2;
    cells[ii + jj* nx].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii + jj* nx].speeds[3] -= w1;
    cells[ii + jj* nx].speeds[6] -= w2;
    cells[ii + jj* nx].speeds[7] -= w2;
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii + jj*nx].speeds[0] = cells[ii + jj*nx].speeds[0]; /* central cell, no movement */
  tmp_cells[ii + jj*nx].speeds[1] = cells[x_w + jj*nx].speeds[1]; /* east */
  tmp_cells[ii + jj*nx].speeds[2] = cells[ii + y_s*nx].speeds[2]; /* north */
  tmp_cells[ii + jj*nx].speeds[3] = cells[x_e + jj*nx].speeds[3]; /* west */
  tmp_cells[ii + jj*nx].speeds[4] = cells[ii + y_n*nx].speeds[4]; /* south */
  tmp_cells[ii + jj*nx].speeds[5] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
  tmp_cells[ii + jj*nx].speeds[6] = cells[x_e + y_s*nx].speeds[6]; /* north-west */
  tmp_cells[ii + jj*nx].speeds[7] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
  tmp_cells[ii + jj*nx].speeds[8] = cells[x_w + y_n*nx].speeds[8]; /* south-east */
}


kernel void rebound(global t_speed* cells ,global t_speed* tmp_cells,
                            global int* obstacles,
                            int nx)
{

  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

      /* if the cell contains an obstacle */
      if (obstacles[jj*nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells[ii + jj*nx].speeds[1] = tmp_cells[ii + jj*nx].speeds[3];
        cells[ii + jj*nx].speeds[2] = tmp_cells[ii + jj*nx].speeds[4];
        cells[ii + jj*nx].speeds[3] = tmp_cells[ii + jj*nx].speeds[1];
        cells[ii + jj*nx].speeds[4] = tmp_cells[ii + jj*nx].speeds[2];
        cells[ii + jj*nx].speeds[5] = tmp_cells[ii + jj*nx].speeds[7];
        cells[ii + jj*nx].speeds[6] = tmp_cells[ii + jj*nx].speeds[8];
        cells[ii + jj*nx].speeds[7] = tmp_cells[ii + jj*nx].speeds[5];
        cells[ii + jj*nx].speeds[8] = tmp_cells[ii + jj*nx].speeds[6];
      }


}

kernel void collision(global t_speed* cells ,global t_speed* tmp_cells,
                            global int* obstacles,
                            int nx, int ny, float omega)
{


  int ii = get_global_id(0);
  int jj = get_global_id(1);


  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

      /* don't consider occupied cells */
      if (!obstacles[ii + jj*nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < 9; kk++)
        {
          local_density += tmp_cells[ii + jj*nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[ii + jj*nx].speeds[1]
                      + tmp_cells[ii + jj*nx].speeds[5]
                      + tmp_cells[ii + jj*nx].speeds[8]
                      - (tmp_cells[ii + jj*nx].speeds[3]
                         + tmp_cells[ii + jj*nx].speeds[6]
                         + tmp_cells[ii + jj*nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[ii + jj*nx].speeds[2]
                      + tmp_cells[ii + jj*nx].speeds[5]
                      + tmp_cells[ii + jj*nx].speeds[6]
                      - (tmp_cells[ii + jj*nx].speeds[4]
                         + tmp_cells[ii + jj*nx].speeds[7]
                         + tmp_cells[ii + jj*nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[9];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[9];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < 9; kk++)
        {
          cells[ii + jj*nx].speeds[kk] = tmp_cells[ii + jj*nx].speeds[kk]
                                                  + omega
                                                  * (d_equ[kk] - tmp_cells[ii + jj*nx].speeds[kk]);
        }
      }







}











kernel void av_vels(global t_speed* cells,
                            global int* obstacles,
                            int nx,
                            int ny,
                            local float* local_u ,
                            global float* partial_u)
{

  // global id's
  int ii = get_global_id(0);
  int jj = get_global_id(1);


  // local data
  int num_wrk_items_x = get_local_size(0);
  int num_wrk_items_y = get_local_size(1);
  int total_work_items = num_wrk_items_x* num_wrk_items_y;

  int local_id_x       = get_local_id(0);
  int local_id_y       = get_local_id(1);

  int group_id_x       = get_group_id(0);
  int group_id_y       = get_group_id(1);

  int num_groups_x = nx/num_wrk_items_x;
  int num_groups_y = ny/num_wrk_items_y;



  // calculate cell index in work group and group index for whole problem
  int cell_index = (num_wrk_items_x * local_id_y) +  local_id_x;
  int group_index = (num_groups_x *group_id_y) +  group_id_x;



  //init local_u for this cell to 0
  local_u[cell_index] = 0.0f;





    // calculate u's
      /* ignore occupied cells */
      if (!obstacles[ii + jj*nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*nx].speeds[1]
                      + cells[ii + jj*nx].speeds[5]
                      + cells[ii + jj*nx].speeds[8]
                      - (cells[ii + jj*nx].speeds[3]
                         + cells[ii + jj*nx].speeds[6]
                         + cells[ii + jj*nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*nx].speeds[2]
                      + cells[ii + jj*nx].speeds[5]
                      + cells[ii + jj*nx].speeds[6]
                      - (cells[ii + jj*nx].speeds[4]
                         + cells[ii + jj*nx].speeds[7]
                         + cells[ii + jj*nx].speeds[8]))
                     / local_density;



        // calculate cell local_u value
          local_u[cell_index] = sqrt((u_x * u_x) + (u_y * u_y));

      }


//REDUCTION (adding up all the totals from within each work group)

work_group_barrier(CLK_LOCAL_MEM_FENCE);

// work group variables
float work_group_total_u = 0.0f;


// Check so you only do the summing on one cell
if (local_id_x == 0 && local_id_y == 0) {

  //init to 0
  work_group_total_u = 0.0f;
  partial_u[group_index] = 0;


  //sum all cells in work group
  for (size_t i=0; i<total_work_items; i++) {
      work_group_total_u += local_u[i];

  }


  //add work group sums to global arrays
  partial_u[group_index] = work_group_total_u;


}










}
