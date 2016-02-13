/*
CUDA TSP solver
Tuomas Rintamäki 2016
tuomas.rintamaki@aalto.fi
*/

/*
License for the helper code for reading the TSPLIB files:

Copyright (c) 2014, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand_kernel.h>

#define dist(a, b) __float2int_rn(sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])))
#define swap(a, b) {int tmp = a;  a = b;  b = tmp;}

static __device__ volatile int best_d;
static __device__ volatile int sol_d;

__global__ void Init()
{
  sol_d = 0;
  best_d = INT_MAX;
}

__global__ void TwoOpt(int cities, float *posx, float *posy, float *px, float *py, int *tour, int *len)
{
  int a,b,c,d, Dab;
  int i,j,ii,jj,mini,minj,from,to,cost,offset;
  int minchange, change;

  offset = blockIdx.x*cities;

  // copy the city coordinates and set the initial tour for each block
  for (i = 0; i < cities; i++) {
    px[offset+i] = posx[i];
    py[offset+i] = posy[i];
    tour[offset+i] = i;
  }

  // do serial permutation of the city coordinates so that initial tour is randomized
  curandState rndstate;
  curand_init(blockIdx.x, 0, 0, &rndstate);
  for (i = 0; i < cities; i++) {
    j = curand(&rndstate) % (cities);
    swap(tour[offset+i], tour[offset+j]);
  }
  
  // search for 2-opt moves
  do {
      minchange = 0;
      i = 0;
      b = tour[offset+cities-1];

      while (i < cities-3) {
          a = b;
          i = i+1;
          b = tour[offset+i];
          Dab = dist(a,b);
          j = i+1;
          d = tour[offset+j];

          while (j < cities-1) {
              c = d;
              j = j+1;
              d = tour[offset+j];

              change = dist(a,c) - dist(c,d) + dist(b,d) - Dab;
              if (change < minchange) {
                  //printf("change: %d \n", change);
                  //printf("minchange: %d \n", minchange);
                  minchange = change;
                  mini = i;
                  minj = j;
              }
          }
      }

      // apply the best move
      if (minchange < 0) {
        i = mini;
        j = minj-1;

        while (i < j) {
          swap(tour[offset+j], tour[offset+i]);
          i++;
          j--;
        }
      }
  } while (minchange < 0);

  // we have a local minimum so compute the cost of the tour
  cost = 0;
  for (i = 0;i < cities - 1;i++) {
    from = tour[offset+i];
    to = tour[offset+i+1];
    cost += dist(from,to);
  }

  // check if the current local minimum is the best solution so far and save it if necessary
  atomicMin((int *)&best_d, cost);
  if (best_d == cost) {
    sol_d = blockIdx.x;
  }
}


/******************************************************************************/
/*** helper code **************************************************************/
/******************************************************************************/

static void CudaTest(char *msg)
{
  cudaError_t e;

  cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}

#define mallocOnGPU(addr, size) if (cudaSuccess != cudaMalloc((void **)&addr, size)) fprintf(stderr, "could not allocate GPU memory\n");  CudaTest("couldn't allocate GPU memory");
#define copyToGPU(to, from, size) if (cudaSuccess != cudaMemcpy(to, from, size, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of data to device failed\n");  CudaTest("data copy to device failed");
#define copyFromGPU(to, from, size) if (cudaSuccess != cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of data from device failed\n");  CudaTest("data copy from device failed");
#define copyFromGPUSymbol(to, from, size) if (cudaSuccess != cudaMemcpyFromSymbol(to, from, size)) fprintf(stderr, "copying of symbol from device failed\n");  CudaTest("symbol copy from device failed");
#define copyToGPUSymbol(to, from, size) if (cudaSuccess != cudaMemcpyToSymbol(to, from, size)) fprintf(stderr, "copying of symbol to device failed\n");  CudaTest("symbol copy to device failed");

/******************************************************************************/
/*** read TSPLIB input ********************************************************/
/******************************************************************************/

static int readInput(char *fname, float **posx_d, float **posy_d)  // ATT and CEIL_2D edge weight types are not supported
{
  int ch, cnt, in1, cities, i, j;
  float in2, in3;
  FILE *f;
  float *posx, *posy;
  char str[256];  // potential for buffer overrun

  f = fopen(fname, "rt");
  if (f == NULL) {fprintf(stderr, "could not open file %s\n", fname);  exit(-1);}

  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

  ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
  fscanf(f, "%s\n", str);
  cities = atoi(str);
  if (cities <= 2) {fprintf(stderr, "only %d cities\n", cities);  exit(-1);}

  posx = (float *)malloc(sizeof(float) * cities);  if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
  posy = (float *)malloc(sizeof(float) * cities);  if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}

  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  fscanf(f, "%s\n", str);
  if (strcmp(str, "NODE_COORD_SECTION") != 0) {fprintf(stderr, "wrong file format\n");  exit(-1);}

  cnt = 0;
  while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) {
    posx[cnt] = in2;
    posy[cnt] = in3;
    cnt++;
    if (cnt > cities) {fprintf(stderr, "input too long\n");  exit(-1);}
    if (cnt != in1) {fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);  exit(-1);}
  }
  if (cnt != cities) {fprintf(stderr, "read %d instead of %d cities\n", cnt, cities);  exit(-1);}

  fscanf(f, "%s", str);
  if (strcmp(str, "EOF") != 0) {fprintf(stderr, "didn't see 'EOF' at end of file\n");  exit(-1);}

  mallocOnGPU(*posx_d, sizeof(float) * cities);
  mallocOnGPU(*posy_d, sizeof(float) * cities);
  copyToGPU(*posx_d, posx, sizeof(float) * cities);
  copyToGPU(*posy_d, posy, sizeof(float) * cities);

  fclose(f);
  free(posx);
  free(posy);

  return cities;
}

int main(int argc, char *argv[])
{
  printf("2-opt TSP CUDA GPU code v0.001 \n");

  int cities, restarts, climbs, best, sol;
  int *tour;
  int *tour_d, *len_d;
  float *posx_d, *posy_d, *px_d, *py_d;
  double runtime;
  struct timeval starttime, endtime;

  if (argc != 3) {fprintf(stderr, "\narguments: input_file restart_count\n"); exit(-1);}
  cities = readInput(argv[1], &posx_d, &posy_d);
  restarts = atoi(argv[2]);
  if (restarts < 1) {fprintf(stderr, "restart_count is too small: %d\n", restarts); exit(-1);}

  printf("configuration: %d cities, %d restarts, %s input\n", cities, restarts, argv[1]);

  cudaFuncSetCacheConfig(TwoOpt, cudaFuncCachePreferEqual);

  // allocate memory for saving blockwise x and y positions on the device as well as the tour orders
  mallocOnGPU(px_d, restarts*cities*sizeof(float));
  mallocOnGPU(py_d, restarts*cities*sizeof(float));
  mallocOnGPU(tour_d, restarts*cities*sizeof(int));
  // also, allocate memory for saving the blockwise tour lengths and the final solution
  mallocOnGPU(len_d, restarts*sizeof(int));
  mallocOnGPU(sol_d, cities*sizeof(int));

  gettimeofday(&starttime, NULL);
  Init<<<1, 1>>>();
  TwoOpt<<<restarts, 1>>>(cities, posx_d, posy_d, px_d, py_d, tour_d, len_d);
  CudaTest("kernel launch failed");
  gettimeofday(&endtime, NULL);
  runtime = endtime.tv_sec + endtime.tv_usec / 1000000.0 - starttime.tv_sec - starttime.tv_usec / 1000000.0;

  // read results
  copyFromGPUSymbol(&best, best_d, sizeof(int));
  copyFromGPUSymbol(&sol, sol_d, sizeof(int));
  tour = (int *)malloc(sizeof(int)*cities);  if (tour == NULL) {fprintf(stderr, "cannot allocate tour\n");  exit(-1);}
  copyFromGPU(tour, &tour_d[sol*cities], sizeof(int)*cities);

  // output results
  printf("best instance = %d \n", sol);
  printf("best found tour length = %d\n", best);
  for (int i = 0; i < cities; i++) {
    printf("node %d \n", tour[i]);
  }

  fflush(stdout);

  cudaFree(posx_d);
  cudaFree(posy_d);
  return 0;
}
