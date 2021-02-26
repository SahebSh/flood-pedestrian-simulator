/*
 * Copyright 2018 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

#include "header.h"
#include "FloodMap.h"

// Defines for Z and H range
#define Z0_RANGE (Z0_MAX - Z0_MIN)
#define H_RANGE (H_MAX - H_MIN)

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
}

//KERNEL DEFINITIONS
/** output_floodmap_to_TBO
 * Outputs flood cell agent data from FLAME GPU to a 4 component vector used for instancing
 * @param	agents	flood cell agent list from FLAME GPU
 * @param	data four component vector used to output instance data 
 */
__global__ void output_floodmap_to_TBO(xmachine_memory_FloodCell_list* agents, glm::vec4* data, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	data[index].x = agents->x[index];// -centralise.x;
	data[index].y = agents->y[index];// -centralise.y;
	data[index].z =  1.0f - ((agents->z0[index] - Z0_MIN) / Z0_RANGE);
	data[index].w = ((agents->h[index] - H_MIN) / H_RANGE); //inverse
}

//EXTERNAL FUNCTIONS DEFINED IN FloodMap.h
extern void generate_instances(cudaGraphicsResource_t * instances_cgr)
{
	//kernals sizes
	int threads_per_tile = 128;
	int tile_size;
	dim3 grid;
    dim3 threads;

	//pointer
	glm::vec4 *dptr_1;
	
	if (get_agent_FloodCell_Default_count() > 0)
	{
		//centralise
		//discrete variables
		int population_width = (int)floor(sqrt((float)get_agent_FloodCell_Default_count()));
		glm::vec3 centralise;
		centralise.x = population_width / 2.0;
		centralise.y = population_width / 2.0;
		centralise.z = 0.0;

		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, instances_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr_1, 0, *instances_cgr));

		//cuda block size
		tile_size = (int) ceil((float)get_agent_FloodCell_Default_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
		//call kernel
		output_floodmap_to_TBO<<< grid, threads>>>(get_device_FloodCell_Default_agents(), dptr_1, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, instances_cgr));
	}
}

