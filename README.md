# cu-tsp
CUDA implementation of the 2-opt heuristic for the travelling salesman problem (TSP).

# Compilation and running the solver
```
nvcc -O3 -arch=sm_50 -use_fast_math tsp.cu -o tsp
./tsp problem.tsp n
```
where problem.tsp is a problem instance from the TSPLIB and n the number of random restarts. On my Nvidia Quadro K2200, the parameter n can take values from 1 to 1024.
