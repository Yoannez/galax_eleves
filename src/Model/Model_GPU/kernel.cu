#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n_particles){
		for(int j = 0; j < n_particles; j++){
			if(i != j)
			{
				const float diffx = positionsGPU[j].x - positionsGPU[i].x;
				const float diffy = positionsGPU[j].y - positionsGPU[i].y;
				const float diffz = positionsGPU[j].z - positionsGPU[i].z;
				
				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < EPS)
				{
					dij = 10.0;
				}
				else
				{
					dij = rsqrtf(dij);
					dij = 10.0 * (dij * dij * dij);
				}
				accelerationsGPU[i].x += diffx * dij * massesGPU[j];
				accelerationsGPU[i].y += diffy * dij * massesGPU[j];
				accelerationsGPU[i].z += diffz * dij * massesGPU[j];
			}
		}
	}
	
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;

	positionsGPU[i].x += velocitiesGPU[i].x * DIFF_T ;
	positionsGPU[i].y += velocitiesGPU[i].y * DIFF_T ;
	positionsGPU[i].z += velocitiesGPU[i].z * DIFF_T ;
}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU);
}


#endif // GALAX_MODEL_GPU