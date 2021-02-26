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
#ifndef _FLOODMAP
#define _FLOODMAP

#include <cuda_gl_interop.h>
#include "CustomVisualisation.h"

enum FLOODMAP_VIEW_MODE { FLOODMAP_VIEW_WATER, FLOODMAP_VIEW_Z0 };
typedef enum FLOODMAP_VIEW_MODE FLOODMAP_VIEW_MODE;

/** initFloodMap
 * Initialises the flood Map by loading model data and creating appropriate buffer objects and shaders
 */
void initFloodMap();

/** renderFloodMap
 * Renders the flood Map by outputting agent data to a texture buffer object and then using vertex texture instancing 
 */
void renderFloodMap();

/** setFloodMapOnOff
 * Turns the flood map display on or off
 * @param state on off state 
 */
void setFloodMapOnOff(TOGGLE_STATE state);

/** toggleFloodMapOnOff
 * Toggles the flood map on or off
 */
void toggleFloodMapOnOff();

/**
* Sets the flood map display to either water (H) or bed level height (Z0)
*/
void setFloodMapDisplayMode(FLOODMAP_VIEW_MODE);


//EXTERNAL FUNCTIONS IMPLEMENTD IN FloodMap.cu CUDA FILE
/** generate_instances
 *  Generates instances by calling a CUDA Kernel which outputs agent data to a texture buffer object
 * @param instances_tbo Texture Buffer Object used for storing instances data
 */
extern void generate_instances(cudaGraphicsResource_t * instances_cgr);

// flood model specific ranges
#define Z0_MIN 0.0f
#define Z0_MAX 40 //1.0f // 2.0f //40 //2.0f //

#define H_MIN 0.0f
#define H_MAX 0.02f // 0.01f

//Circle model fidelity
const int SPHERE_SLICES = 8;
const int SPHERE_STACKS = 8;
const double SPHERE_RADIUS = 0.02;

/** Vertex Shader source for rendering directional arrows */
static const char floodmap_vshader_source[] =
{
	"#extension GL_EXT_gpu_shader4 : enable										\n"
	"uniform samplerBuffer displacementMap;										\n"
	"attribute in float mapIndex;												\n"
	"uniform bool water;														\n"
	"uniform float FM_WIDTH;													\n"
	"uniform float ENV_MAX;														\n"
	"uniform float ENV_WIDTH;													\n"
	"void main()																\n"
	"{																			\n"
	"	vec4 position = gl_Vertex;											    \n"
	"	vec4 lookup = texelFetchBuffer(displacementMap, (int)mapIndex);		    \n"
	"	//if (water)	               											\n"
	"		gl_FrontColor = vec4(1.0f-lookup.w, 1.0f-lookup.w, 1.0f, 0.0);	\n"
	"	//else	                												\n"
	"		gl_FrontColor += vec4(lookup.z, lookup.z, lookup.z, 0.0);			\n"
	"	//lookup.w = 1.0;												    	\n"
	"   //offset model position													\n"
	"	float x_displace = ((lookup.x+0.5)/(FM_WIDTH/ENV_WIDTH))-ENV_MAX;		\n"
	"	float y_displace = ((lookup.y+0.5)/(FM_WIDTH/ENV_WIDTH))-ENV_MAX;		\n"
	"   position.x += x_displace;												\n"
	"   position.y += y_displace;												\n"
	"   position.z += 0.05;												\n"

	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    		\n"
	"}																			\n"
};

#endif //_FLOODMAP
