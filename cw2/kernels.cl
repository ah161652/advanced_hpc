#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

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


kernel void fusion1(global float* speeds0,
                    global float* speeds1,
                    global float* speeds2,
                    global float* speeds3,
                    global float* speeds4,
                    global float* speeds5,
                    global float* speeds6,
                    global float* speeds7,
                    global float* speeds8,
                            global float* tmp_speeds0,
                            global float* tmp_speeds1,
                            global float* tmp_speeds2,
                            global float* tmp_speeds3,
                            global float* tmp_speeds4,
                            global float* tmp_speeds5,
                            global float* tmp_speeds6,
                            global float* tmp_speeds7,
                            global float* tmp_speeds8,
                            global int* obstacles,
                            int nx,
                            int ny,
                            local float* local_u ,
                            global float* partial_u,
                            float omega, float accel, float density)
{

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  const float w4 = 2.f * c_sq;
  const float w3 =w4 * c_sq;

  float tot_u =0.f;         /* accumulated magnitudes of velocity for each cell */

  int ii = get_global_id(0);
  int jj = get_global_id(1);










  /* compute weighting factors */
  float a1 = density * accel / 9.0;
  float a2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jjj = ny - 2;


  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jjj* nx]
    && (speeds3[ii + jj*nx] - a1) > 0.f
     && (speeds6[ii + jj*nx] - a2) > 0.f
     && (speeds7[ii + jj*nx] - a2) > 0.f
      && jj==0)
  {
    /* increase 'east-side' densities */
    speeds1[ii + jj*nx] += a1;
    speeds5[ii + jj*nx] += a2;
    speeds8[ii + jj*nx] += a2;
    /* decrease 'west-side' densities */
    speeds3[ii + jj*nx] -= a1;
    speeds6[ii + jj*nx] -= a2;
    speeds7[ii + jj*nx] -= a2;
  }


    work_group_barrier(CLK_LOCAL_MEM_FENCE);


  ///////////////////////////////////
  //////////////////////////////////
  /////////////FUSIONED LOOP//////
  //////////////////////////////////
  /////////////////////////////////


  //PROPAGATE VARIABLES
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);



  /* compute local density total */
  float local_density = 0.f;

  local_density = local_density + speeds0[ii + jj*nx] + speeds1[x_w + jj*nx] + speeds2[ii + y_s*nx] + speeds3[x_e + jj*nx] + speeds4[ii + y_n*nx] + speeds5[x_w + y_s*nx] + speeds6[x_e + y_s*nx] + speeds7[x_e + y_n*nx] + speeds8[x_w + y_n*nx];




  /* compute x velocity component */
  float u_x = (speeds1[x_w + jj*nx]
                + speeds5[x_w + y_s*nx]
                + speeds8[x_w + y_n*nx]
                - (speeds3[x_e + jj*nx]
                   + speeds6[x_e + y_s*nx]
                   + speeds7[x_e + y_n*nx]))
               / local_density;

  /* compute y velocity component */
  float u_y = (speeds2[ii + y_s*nx]
                + speeds5[x_w + y_s*nx]
                + speeds6[x_e + y_s*nx]
                - (speeds4[ii + y_n*nx]
                   + speeds7[x_e + y_n*nx]
                   +speeds8[x_w + y_n*nx]))
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


  #pragma novector
  d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq + (u[1]*u[1]) / w3 - u_sq / w4);
  d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq + (u[2]*u[2]) / w3 - u_sq / w4);
  d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq + (u[3]*u[3]) / w3 - u_sq / w4);
  d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq + (u[4]*u[4]) / w3 - u_sq / w4);

  d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq + (u[5]*u[5]) / w3 - u_sq / w4);
  d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq + (u[6]*u[6]) / w3 - u_sq / w4);
  d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq + (u[7]*u[7]) / w3 - u_sq / w4);
  d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq + (u[8]*u[8]) / w3 - u_sq / w4);



  tmp_speeds0[ii + jj*nx] = (!obstacles[jj*nx + ii]) ? speeds0[ii + jj*nx] + omega * (d_equ[0] - speeds0[ii + jj*nx]) :   tmp_speeds0[ii + jj*nx];
  tmp_speeds1[ii + jj*nx] = (obstacles[jj*nx + ii]) ? speeds3[x_e + jj*nx] : speeds1[x_w + jj*nx]  + omega * (d_equ[1] - speeds1[x_w + jj*nx] );
  tmp_speeds2[ii + jj*nx] = (obstacles[jj*nx + ii]) ? speeds4[ii + y_n*nx] : speeds2[ii + y_s*nx]  + omega * (d_equ[2] - speeds2[ii + y_s*nx] );
  tmp_speeds3[ii + jj*nx] = (obstacles[jj*nx + ii]) ? speeds1[x_w + jj*nx] : speeds3[x_e + jj*nx] + omega * (d_equ[3] -speeds3[x_e + jj*nx] );
  tmp_speeds4[ii + jj*nx] = (obstacles[jj*nx + ii]) ? speeds2[ii + y_s*nx] : speeds4[ii + y_n*nx] + omega * (d_equ[4] -speeds4[ii + y_n*nx] );
  tmp_speeds5[ii + jj*nx] = (obstacles[jj*nx + ii]) ? speeds7[x_e + y_n*nx] : speeds5[x_w + y_s*nx] + omega * (d_equ[5] -speeds5[x_w + y_s*nx] );
  tmp_speeds6[ii + jj*nx] = (obstacles[jj*nx + ii]) ? speeds8[x_w + y_n*nx] : speeds6[x_e + y_s*nx] + omega * (d_equ[6] -speeds6[x_e + y_s*nx] );
  tmp_speeds7[ii + jj*nx] = (obstacles[jj*nx + ii]) ? speeds5[x_w + y_s*nx] : speeds7[x_e + y_n*nx] + omega * (d_equ[7] - speeds7[x_e + y_n*nx]);
  tmp_speeds8[ii + jj*nx] = (obstacles[jj*nx + ii]) ? speeds6[x_e + y_s*nx] : speeds8[x_w + y_n*nx] + omega * (d_equ[8] -speeds8[x_w + y_n*nx] );



  ///////////////////////////////////
  //////////////////////////////////
  /////////////AV  VELS//////
  //////////////////////////////////
  /////////////////////////////////

  work_group_barrier(CLK_LOCAL_MEM_FENCE);





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


  // calculate cell local_u value
    local_u[cell_index] = sqrt((u_x * u_x) + (u_y * u_y));


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



kernel void fusion2(global float* speeds0,
                    global float* speeds1,
                    global float* speeds2,
                    global float* speeds3,
                    global float* speeds4,
                    global float* speeds5,
                    global float* speeds6,
                    global float* speeds7,
                    global float* speeds8,
                            global float* tmp_speeds0,
                            global float* tmp_speeds1,
                            global float* tmp_speeds2,
                            global float* tmp_speeds3,
                            global float* tmp_speeds4,
                            global float* tmp_speeds5,
                            global float* tmp_speeds6,
                            global float* tmp_speeds7,
                            global float* tmp_speeds8,
                            global int* obstacles,
                            int nx,
                            int ny,
                            local float* local_u ,
                            global float* partial_u,
                            float omega, float accel, float density)
{




  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  const float w4 = 2.f * c_sq;
  const float w3 =w4 * c_sq;

  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u =0.f;         /* accumulated magnitudes of velocity for each cell */

  int ii = get_global_id(0);
  int jj = get_global_id(1);





  /* compute weighting factors */
  float a1 = density * accel / 9.0;
  float a2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jjj = ny - 2;


  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jjj* nx]
    && (tmp_speeds3[ii + jj*nx] - a1) > 0.f
      && (tmp_speeds6[ii + jj*nx] - a2) > 0.f
      && (tmp_speeds7[ii + jj*nx] - a2) > 0.f
      && jj==0)
  {
    /* increase 'east-side' densities */
    tmp_speeds1[ii + jj*nx] += a1;
      tmp_speeds5[ii + jj*nx] += a2;
      tmp_speeds8[ii + jj*nx] += a2;
      /* decrease 'west-side' densities */
      tmp_speeds3[ii + jj*nx] -= a1;
      tmp_speeds6[ii + jj*nx] -= a2;
      tmp_speeds7[ii + jj*nx] -= a2;
  }


    work_group_barrier(CLK_LOCAL_MEM_FENCE);




  ///////////////////////////////////
  //////////////////////////////////
  /////////////FUSIONED LOOP//////
  //////////////////////////////////
  /////////////////////////////////



  //PROPAGATE VARIABLES
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);



  /* compute local density total */
  float local_density = 0.f;

  local_density = local_density + tmp_speeds0[ii + jj*nx] + tmp_speeds1[x_w + jj*nx] + tmp_speeds2[ii + y_s*nx] + tmp_speeds3[x_e + jj*nx] + tmp_speeds4[ii + y_n*nx] + tmp_speeds5[x_w + y_s*nx] + tmp_speeds6[x_e + y_s*nx] + tmp_speeds7[x_e + y_n*nx] + tmp_speeds8[x_w + y_n*nx];



  /* compute x velocity component */
  float u_x = (tmp_speeds1[x_w + jj*nx]
                + tmp_speeds5[x_w + y_s*nx]
                + tmp_speeds8[x_w + y_n*nx]
                - (tmp_speeds3[x_e + jj*nx]
                   + tmp_speeds6[x_e + y_s*nx]
                   + tmp_speeds7[x_e + y_n*nx]))
               / local_density;

  /* compute y velocity component */
  float u_y = (tmp_speeds2[ii + y_s*nx]
                + tmp_speeds5[x_w + y_s*nx]
                + tmp_speeds6[x_e + y_s*nx]
                - (tmp_speeds4[ii + y_n*nx]
                   + tmp_speeds7[x_e + y_n*nx]
                   +tmp_speeds8[x_w + y_n*nx]))
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


  #pragma novector
  d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq + (u[1]*u[1]) / w3 - u_sq / w4);
  d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq + (u[2]*u[2]) / w3 - u_sq / w4);
  d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq + (u[3]*u[3]) / w3 - u_sq / w4);
  d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq + (u[4]*u[4]) / w3 - u_sq / w4);

  d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq + (u[5]*u[5]) / w3 - u_sq / w4);
  d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq + (u[6]*u[6]) / w3 - u_sq / w4);
  d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq + (u[7]*u[7]) / w3 - u_sq / w4);
  d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq + (u[8]*u[8]) / w3 - u_sq / w4);



        speeds0[ii + jj*nx] = (!obstacles[jj*nx + ii]) ? tmp_speeds0[ii + jj*nx] + omega * (d_equ[0] - tmp_speeds0[ii + jj*nx]) :   speeds0[ii + jj*nx];
        speeds1[ii + jj*nx] = (obstacles[jj*nx + ii]) ? tmp_speeds3[x_e + jj*nx] : tmp_speeds1[x_w + jj*nx]  + omega * (d_equ[1] - tmp_speeds1[x_w + jj*nx] );
        speeds2[ii + jj*nx] = (obstacles[jj*nx + ii]) ? tmp_speeds4[ii + y_n*nx] : tmp_speeds2[ii + y_s*nx]  + omega * (d_equ[2] - tmp_speeds2[ii + y_s*nx] );
        speeds3[ii + jj*nx] = (obstacles[jj*nx + ii]) ? tmp_speeds1[x_w + jj*nx] : tmp_speeds3[x_e + jj*nx] + omega * (d_equ[3] -tmp_speeds3[x_e + jj*nx] );
        speeds4[ii + jj*nx] = (obstacles[jj*nx + ii]) ? tmp_speeds2[ii + y_s*nx] : tmp_speeds4[ii + y_n*nx] + omega * (d_equ[4] -tmp_speeds4[ii + y_n*nx] );
        speeds5[ii + jj*nx] = (obstacles[jj*nx + ii]) ? tmp_speeds7[x_e + y_n*nx] : tmp_speeds5[x_w + y_s*nx] + omega * (d_equ[5] -tmp_speeds5[x_w + y_s*nx] );
        speeds6[ii + jj*nx] = (obstacles[jj*nx + ii]) ? tmp_speeds8[x_w + y_n*nx] : tmp_speeds6[x_e + y_s*nx] + omega * (d_equ[6] -tmp_speeds6[x_e + y_s*nx] );
        speeds7[ii + jj*nx] = (obstacles[jj*nx + ii]) ? tmp_speeds5[x_w + y_s*nx] : tmp_speeds7[x_e + y_n*nx] + omega * (d_equ[7] - tmp_speeds7[x_e + y_n*nx]);
        speeds8[ii + jj*nx] = (obstacles[jj*nx + ii]) ? tmp_speeds6[x_e + y_s*nx] : tmp_speeds8[x_w + y_n*nx] + omega * (d_equ[8] -tmp_speeds8[x_w + y_n*nx] );



  ///////////////////////////////////
  //////////////////////////////////
  /////////////AV  VELS//////
  //////////////////////////////////
  /////////////////////////////////

  work_group_barrier(CLK_LOCAL_MEM_FENCE);





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


  // calculate cell local_u value
    local_u[cell_index] = sqrt((u_x * u_x) + (u_y * u_y));


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
