
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
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



#ifndef __HEADER
#define __HEADER

#if defined __NVCC__
   // Disable annotation on defaulted function warnings (glm 0.9.9 and CUDA 9.0 introduced this warning)
   #pragma diag_suppress esa_on_defaulted_function_ignored 
#endif

#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__
#define __FLAME_GPU_HOST_FUNC__ __host__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

// FLAME GPU Version Macros.
#define FLAME_GPU_MAJOR_VERSION 1
#define FLAME_GPU_MINOR_VERSION 5
#define FLAME_GPU_PATCH_VERSION 0

typedef unsigned int uint;

//FLAME GPU vector types float, (i)nteger, (u)nsigned integer, (d)ouble
typedef glm::vec2 fvec2;
typedef glm::vec3 fvec3;
typedef glm::vec4 fvec4;
typedef glm::ivec2 ivec2;
typedef glm::ivec3 ivec3;
typedef glm::ivec4 ivec4;
typedef glm::uvec2 uvec2;
typedef glm::uvec3 uvec3;
typedef glm::uvec4 uvec4;
typedef glm::dvec2 dvec2;
typedef glm::dvec3 dvec3;
typedef glm::dvec4 dvec4;

	
//if this is defined then the project must be built with sm_13 or later
#define _DOUBLE_SUPPORT_REQUIRED_

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 16384

//Maximum population size of xmachine_memory_FloodCell
#define xmachine_memory_FloodCell_MAX 16384

//Maximum population size of xmachine_memory_agent
#define xmachine_memory_agent_MAX 16384

//Maximum population size of xmachine_memory_navmap
#define xmachine_memory_navmap_MAX 16384


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_WetDryMessage
#define xmachine_message_WetDryMessage_MAX 16384

//Maximum population size of xmachine_mmessage_SpaceOperatorMessage
#define xmachine_message_SpaceOperatorMessage_MAX 16384

//Maximum population size of xmachine_mmessage_FloodData
#define xmachine_message_FloodData_MAX 16384

//Maximum population size of xmachine_mmessage_pedestrian_location
#define xmachine_message_pedestrian_location_MAX 16384

//Maximum population size of xmachine_mmessage_PedData
#define xmachine_message_PedData_MAX 16384

//Maximum population size of xmachine_mmessage_updatedNavmapData
#define xmachine_message_updatedNavmapData_MAX 16384

//Maximum population size of xmachine_mmessage_NavmapData
#define xmachine_message_NavmapData_MAX 16384

//Maximum population size of xmachine_mmessage_navmap_cell
#define xmachine_message_navmap_cell_MAX 16384


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_WetDryMessage_partitioningDiscrete
#define xmachine_message_SpaceOperatorMessage_partitioningDiscrete
#define xmachine_message_FloodData_partitioningDiscrete
#define xmachine_message_pedestrian_location_partitioningSpatial
#define xmachine_message_PedData_partitioningSpatial
#define xmachine_message_updatedNavmapData_partitioningDiscrete
#define xmachine_message_NavmapData_partitioningDiscrete
#define xmachine_message_navmap_cell_partitioningDiscrete

/* Spatial partitioning grid size definitions */
//xmachine_message_pedestrian_location partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_pedestrian_location_grid_size 6400
//xmachine_message_PedData partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_PedData_grid_size 160000

/* Static Graph size definitions*/
  

/* Default visualisation Colour indices */
 
#define FLAME_GPU_VISUALISATION_COLOUR_BLACK 0
#define FLAME_GPU_VISUALISATION_COLOUR_RED 1
#define FLAME_GPU_VISUALISATION_COLOUR_GREEN 2
#define FLAME_GPU_VISUALISATION_COLOUR_BLUE 3
#define FLAME_GPU_VISUALISATION_COLOUR_YELLOW 4
#define FLAME_GPU_VISUALISATION_COLOUR_CYAN 5
#define FLAME_GPU_VISUALISATION_COLOUR_MAGENTA 6
#define FLAME_GPU_VISUALISATION_COLOUR_WHITE 7
#define FLAME_GPU_VISUALISATION_COLOUR_BROWN 8

/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_FloodCell
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_FloodCell
{
    int inDomain;    /**< X-machine memory variable inDomain of type int.*/
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
    double z0;    /**< X-machine memory variable z0 of type double.*/
    double h;    /**< X-machine memory variable h of type double.*/
    double qx;    /**< X-machine memory variable qx of type double.*/
    double qy;    /**< X-machine memory variable qy of type double.*/
    double timeStep;    /**< X-machine memory variable timeStep of type double.*/
    double minh_loc;    /**< X-machine memory variable minh_loc of type double.*/
    double hFace_E;    /**< X-machine memory variable hFace_E of type double.*/
    double etFace_E;    /**< X-machine memory variable etFace_E of type double.*/
    double qxFace_E;    /**< X-machine memory variable qxFace_E of type double.*/
    double qyFace_E;    /**< X-machine memory variable qyFace_E of type double.*/
    double hFace_W;    /**< X-machine memory variable hFace_W of type double.*/
    double etFace_W;    /**< X-machine memory variable etFace_W of type double.*/
    double qxFace_W;    /**< X-machine memory variable qxFace_W of type double.*/
    double qyFace_W;    /**< X-machine memory variable qyFace_W of type double.*/
    double hFace_N;    /**< X-machine memory variable hFace_N of type double.*/
    double etFace_N;    /**< X-machine memory variable etFace_N of type double.*/
    double qxFace_N;    /**< X-machine memory variable qxFace_N of type double.*/
    double qyFace_N;    /**< X-machine memory variable qyFace_N of type double.*/
    double hFace_S;    /**< X-machine memory variable hFace_S of type double.*/
    double etFace_S;    /**< X-machine memory variable etFace_S of type double.*/
    double qxFace_S;    /**< X-machine memory variable qxFace_S of type double.*/
    double qyFace_S;    /**< X-machine memory variable qyFace_S of type double.*/
    double nm_rough;    /**< X-machine memory variable nm_rough of type double.*/
};

/** struct xmachine_memory_agent
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_agent
{
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float velx;    /**< X-machine memory variable velx of type float.*/
    float vely;    /**< X-machine memory variable vely of type float.*/
    float steer_x;    /**< X-machine memory variable steer_x of type float.*/
    float steer_y;    /**< X-machine memory variable steer_y of type float.*/
    float height;    /**< X-machine memory variable height of type float.*/
    int exit_no;    /**< X-machine memory variable exit_no of type int.*/
    float speed;    /**< X-machine memory variable speed of type float.*/
    int lod;    /**< X-machine memory variable lod of type int.*/
    float animate;    /**< X-machine memory variable animate of type float.*/
    int animate_dir;    /**< X-machine memory variable animate_dir of type int.*/
    int HR_state;    /**< X-machine memory variable HR_state of type int.*/
    int hero_status;    /**< X-machine memory variable hero_status of type int.*/
    double pickup_time;    /**< X-machine memory variable pickup_time of type double.*/
    double drop_time;    /**< X-machine memory variable drop_time of type double.*/
    int carry_sandbag;    /**< X-machine memory variable carry_sandbag of type int.*/
    double HR;    /**< X-machine memory variable HR of type double.*/
    float dt_ped;    /**< X-machine memory variable dt_ped of type float.*/
    float d_water;    /**< X-machine memory variable d_water of type float.*/
    float v_water;    /**< X-machine memory variable v_water of type float.*/
    float body_height;    /**< X-machine memory variable body_height of type float.*/
    float body_mass;    /**< X-machine memory variable body_mass of type float.*/
    int gender;    /**< X-machine memory variable gender of type int.*/
    int stability_state;    /**< X-machine memory variable stability_state of type int.*/
    float motion_speed;    /**< X-machine memory variable motion_speed of type float.*/
    int age;    /**< X-machine memory variable age of type int.*/
    float excitement_speed;    /**< X-machine memory variable excitement_speed of type float.*/
    int dir_times;    /**< X-machine memory variable dir_times of type int.*/
    int rejected_exit1;    /**< X-machine memory variable rejected_exit1 of type int.*/
    int rejected_exit2;    /**< X-machine memory variable rejected_exit2 of type int.*/
    int rejected_exit3;    /**< X-machine memory variable rejected_exit3 of type int.*/
    int rejected_exit4;    /**< X-machine memory variable rejected_exit4 of type int.*/
    int rejected_exit5;    /**< X-machine memory variable rejected_exit5 of type int.*/
};

/** struct xmachine_memory_navmap
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_navmap
{
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
    double z0;    /**< X-machine memory variable z0 of type double.*/
    double h;    /**< X-machine memory variable h of type double.*/
    double qx;    /**< X-machine memory variable qx of type double.*/
    double qy;    /**< X-machine memory variable qy of type double.*/
    int exit_no;    /**< X-machine memory variable exit_no of type int.*/
    float height;    /**< X-machine memory variable height of type float.*/
    float collision_x;    /**< X-machine memory variable collision_x of type float.*/
    float collision_y;    /**< X-machine memory variable collision_y of type float.*/
    float exit0_x;    /**< X-machine memory variable exit0_x of type float.*/
    float exit0_y;    /**< X-machine memory variable exit0_y of type float.*/
    float exit1_x;    /**< X-machine memory variable exit1_x of type float.*/
    float exit1_y;    /**< X-machine memory variable exit1_y of type float.*/
    float exit2_x;    /**< X-machine memory variable exit2_x of type float.*/
    float exit2_y;    /**< X-machine memory variable exit2_y of type float.*/
    float exit3_x;    /**< X-machine memory variable exit3_x of type float.*/
    float exit3_y;    /**< X-machine memory variable exit3_y of type float.*/
    float exit4_x;    /**< X-machine memory variable exit4_x of type float.*/
    float exit4_y;    /**< X-machine memory variable exit4_y of type float.*/
    float exit5_x;    /**< X-machine memory variable exit5_x of type float.*/
    float exit5_y;    /**< X-machine memory variable exit5_y of type float.*/
    float exit6_x;    /**< X-machine memory variable exit6_x of type float.*/
    float exit6_y;    /**< X-machine memory variable exit6_y of type float.*/
    float exit7_x;    /**< X-machine memory variable exit7_x of type float.*/
    float exit7_y;    /**< X-machine memory variable exit7_y of type float.*/
    float exit8_x;    /**< X-machine memory variable exit8_x of type float.*/
    float exit8_y;    /**< X-machine memory variable exit8_y of type float.*/
    float exit9_x;    /**< X-machine memory variable exit9_x of type float.*/
    float exit9_y;    /**< X-machine memory variable exit9_y of type float.*/
    int drop_point;    /**< X-machine memory variable drop_point of type int.*/
    int sandbag_capacity;    /**< X-machine memory variable sandbag_capacity of type int.*/
    double nm_rough;    /**< X-machine memory variable nm_rough of type double.*/
    int evac_counter;    /**< X-machine memory variable evac_counter of type int.*/
};



/* Message structures */

/** struct xmachine_message_WetDryMessage
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_WetDryMessage
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int inDomain;        /**< Message variable inDomain of type int.*/  
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    double min_hloc;        /**< Message variable min_hloc of type double.*/
};

/** struct xmachine_message_SpaceOperatorMessage
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_SpaceOperatorMessage
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int inDomain;        /**< Message variable inDomain of type int.*/  
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    double hFace_E;        /**< Message variable hFace_E of type double.*/  
    double etFace_E;        /**< Message variable etFace_E of type double.*/  
    double qFace_X_E;        /**< Message variable qFace_X_E of type double.*/  
    double qFace_Y_E;        /**< Message variable qFace_Y_E of type double.*/  
    double hFace_W;        /**< Message variable hFace_W of type double.*/  
    double etFace_W;        /**< Message variable etFace_W of type double.*/  
    double qFace_X_W;        /**< Message variable qFace_X_W of type double.*/  
    double qFace_Y_W;        /**< Message variable qFace_Y_W of type double.*/  
    double hFace_N;        /**< Message variable hFace_N of type double.*/  
    double etFace_N;        /**< Message variable etFace_N of type double.*/  
    double qFace_X_N;        /**< Message variable qFace_X_N of type double.*/  
    double qFace_Y_N;        /**< Message variable qFace_Y_N of type double.*/  
    double hFace_S;        /**< Message variable hFace_S of type double.*/  
    double etFace_S;        /**< Message variable etFace_S of type double.*/  
    double qFace_X_S;        /**< Message variable qFace_X_S of type double.*/  
    double qFace_Y_S;        /**< Message variable qFace_Y_S of type double.*/
};

/** struct xmachine_message_FloodData
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_FloodData
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int inDomain;        /**< Message variable inDomain of type int.*/  
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    double z0;        /**< Message variable z0 of type double.*/  
    double h;        /**< Message variable h of type double.*/  
    double qx;        /**< Message variable qx of type double.*/  
    double qy;        /**< Message variable qy of type double.*/  
    double nm_rough;        /**< Message variable nm_rough of type double.*/
};

/** struct xmachine_message_pedestrian_location
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_pedestrian_location
{	
    /* Spatial Partitioning Variables */
    glm::ivec3 _relative_cell;    /**< Relative cell position from agent grid cell position range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    glm::ivec3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/
};

/** struct xmachine_message_PedData
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_PedData
{	
    /* Spatial Partitioning Variables */
    glm::ivec3 _relative_cell;    /**< Relative cell position from agent grid cell position range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    glm::ivec3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    int hero_status;        /**< Message variable hero_status of type int.*/  
    int pickup_time;        /**< Message variable pickup_time of type int.*/  
    int drop_time;        /**< Message variable drop_time of type int.*/  
    int exit_no;        /**< Message variable exit_no of type int.*/  
    int carry_sandbag;        /**< Message variable carry_sandbag of type int.*/  
    double body_height;        /**< Message variable body_height of type double.*/
};

/** struct xmachine_message_updatedNavmapData
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_updatedNavmapData
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    double z0;        /**< Message variable z0 of type double.*/  
    int drop_point;        /**< Message variable drop_point of type int.*/  
    int sandbag_capacity;        /**< Message variable sandbag_capacity of type int.*/  
    int exit_no;        /**< Message variable exit_no of type int.*/
};

/** struct xmachine_message_NavmapData
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_NavmapData
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    double z0;        /**< Message variable z0 of type double.*/  
    double nm_rough;        /**< Message variable nm_rough of type double.*/
};

/** struct xmachine_message_navmap_cell
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_navmap_cell
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    double z0;        /**< Message variable z0 of type double.*/  
    double h;        /**< Message variable h of type double.*/  
    double qx;        /**< Message variable qx of type double.*/  
    double qy;        /**< Message variable qy of type double.*/  
    int exit_no;        /**< Message variable exit_no of type int.*/  
    float height;        /**< Message variable height of type float.*/  
    float collision_x;        /**< Message variable collision_x of type float.*/  
    float collision_y;        /**< Message variable collision_y of type float.*/  
    float exit0_x;        /**< Message variable exit0_x of type float.*/  
    float exit0_y;        /**< Message variable exit0_y of type float.*/  
    float exit1_x;        /**< Message variable exit1_x of type float.*/  
    float exit1_y;        /**< Message variable exit1_y of type float.*/  
    float exit2_x;        /**< Message variable exit2_x of type float.*/  
    float exit2_y;        /**< Message variable exit2_y of type float.*/  
    float exit3_x;        /**< Message variable exit3_x of type float.*/  
    float exit3_y;        /**< Message variable exit3_y of type float.*/  
    float exit4_x;        /**< Message variable exit4_x of type float.*/  
    float exit4_y;        /**< Message variable exit4_y of type float.*/  
    float exit5_x;        /**< Message variable exit5_x of type float.*/  
    float exit5_y;        /**< Message variable exit5_y of type float.*/  
    float exit6_x;        /**< Message variable exit6_x of type float.*/  
    float exit6_y;        /**< Message variable exit6_y of type float.*/  
    float exit7_x;        /**< Message variable exit7_x of type float.*/  
    float exit7_y;        /**< Message variable exit7_y of type float.*/  
    float exit8_x;        /**< Message variable exit8_x of type float.*/  
    float exit8_y;        /**< Message variable exit8_y of type float.*/  
    float exit9_x;        /**< Message variable exit9_x of type float.*/  
    float exit9_y;        /**< Message variable exit9_y of type float.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_FloodCell_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_FloodCell_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_FloodCell_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_FloodCell_MAX];  /**< Used during parallel prefix sum */
    
    int inDomain [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list inDomain of type int.*/
    int x [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list y of type int.*/
    double z0 [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list z0 of type double.*/
    double h [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list h of type double.*/
    double qx [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qx of type double.*/
    double qy [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qy of type double.*/
    double timeStep [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list timeStep of type double.*/
    double minh_loc [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list minh_loc of type double.*/
    double hFace_E [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list hFace_E of type double.*/
    double etFace_E [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list etFace_E of type double.*/
    double qxFace_E [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qxFace_E of type double.*/
    double qyFace_E [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qyFace_E of type double.*/
    double hFace_W [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list hFace_W of type double.*/
    double etFace_W [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list etFace_W of type double.*/
    double qxFace_W [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qxFace_W of type double.*/
    double qyFace_W [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qyFace_W of type double.*/
    double hFace_N [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list hFace_N of type double.*/
    double etFace_N [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list etFace_N of type double.*/
    double qxFace_N [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qxFace_N of type double.*/
    double qyFace_N [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qyFace_N of type double.*/
    double hFace_S [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list hFace_S of type double.*/
    double etFace_S [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list etFace_S of type double.*/
    double qxFace_S [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qxFace_S of type double.*/
    double qyFace_S [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qyFace_S of type double.*/
    double nm_rough [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list nm_rough of type double.*/
};

/** struct xmachine_memory_agent_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_agent_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_agent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_agent_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_memory_agent_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_agent_MAX];    /**< X-machine memory variable list y of type float.*/
    float velx [xmachine_memory_agent_MAX];    /**< X-machine memory variable list velx of type float.*/
    float vely [xmachine_memory_agent_MAX];    /**< X-machine memory variable list vely of type float.*/
    float steer_x [xmachine_memory_agent_MAX];    /**< X-machine memory variable list steer_x of type float.*/
    float steer_y [xmachine_memory_agent_MAX];    /**< X-machine memory variable list steer_y of type float.*/
    float height [xmachine_memory_agent_MAX];    /**< X-machine memory variable list height of type float.*/
    int exit_no [xmachine_memory_agent_MAX];    /**< X-machine memory variable list exit_no of type int.*/
    float speed [xmachine_memory_agent_MAX];    /**< X-machine memory variable list speed of type float.*/
    int lod [xmachine_memory_agent_MAX];    /**< X-machine memory variable list lod of type int.*/
    float animate [xmachine_memory_agent_MAX];    /**< X-machine memory variable list animate of type float.*/
    int animate_dir [xmachine_memory_agent_MAX];    /**< X-machine memory variable list animate_dir of type int.*/
    int HR_state [xmachine_memory_agent_MAX];    /**< X-machine memory variable list HR_state of type int.*/
    int hero_status [xmachine_memory_agent_MAX];    /**< X-machine memory variable list hero_status of type int.*/
    double pickup_time [xmachine_memory_agent_MAX];    /**< X-machine memory variable list pickup_time of type double.*/
    double drop_time [xmachine_memory_agent_MAX];    /**< X-machine memory variable list drop_time of type double.*/
    int carry_sandbag [xmachine_memory_agent_MAX];    /**< X-machine memory variable list carry_sandbag of type int.*/
    double HR [xmachine_memory_agent_MAX];    /**< X-machine memory variable list HR of type double.*/
    float dt_ped [xmachine_memory_agent_MAX];    /**< X-machine memory variable list dt_ped of type float.*/
    float d_water [xmachine_memory_agent_MAX];    /**< X-machine memory variable list d_water of type float.*/
    float v_water [xmachine_memory_agent_MAX];    /**< X-machine memory variable list v_water of type float.*/
    float body_height [xmachine_memory_agent_MAX];    /**< X-machine memory variable list body_height of type float.*/
    float body_mass [xmachine_memory_agent_MAX];    /**< X-machine memory variable list body_mass of type float.*/
    int gender [xmachine_memory_agent_MAX];    /**< X-machine memory variable list gender of type int.*/
    int stability_state [xmachine_memory_agent_MAX];    /**< X-machine memory variable list stability_state of type int.*/
    float motion_speed [xmachine_memory_agent_MAX];    /**< X-machine memory variable list motion_speed of type float.*/
    int age [xmachine_memory_agent_MAX];    /**< X-machine memory variable list age of type int.*/
    float excitement_speed [xmachine_memory_agent_MAX];    /**< X-machine memory variable list excitement_speed of type float.*/
    int dir_times [xmachine_memory_agent_MAX];    /**< X-machine memory variable list dir_times of type int.*/
    int rejected_exit1 [xmachine_memory_agent_MAX];    /**< X-machine memory variable list rejected_exit1 of type int.*/
    int rejected_exit2 [xmachine_memory_agent_MAX];    /**< X-machine memory variable list rejected_exit2 of type int.*/
    int rejected_exit3 [xmachine_memory_agent_MAX];    /**< X-machine memory variable list rejected_exit3 of type int.*/
    int rejected_exit4 [xmachine_memory_agent_MAX];    /**< X-machine memory variable list rejected_exit4 of type int.*/
    int rejected_exit5 [xmachine_memory_agent_MAX];    /**< X-machine memory variable list rejected_exit5 of type int.*/
};

/** struct xmachine_memory_navmap_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_navmap_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_navmap_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_navmap_MAX];  /**< Used during parallel prefix sum */
    
    int x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list y of type int.*/
    double z0 [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list z0 of type double.*/
    double h [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list h of type double.*/
    double qx [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list qx of type double.*/
    double qy [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list qy of type double.*/
    int exit_no [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit_no of type int.*/
    float height [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list height of type float.*/
    float collision_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list collision_x of type float.*/
    float collision_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list collision_y of type float.*/
    float exit0_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit0_x of type float.*/
    float exit0_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit0_y of type float.*/
    float exit1_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit1_x of type float.*/
    float exit1_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit1_y of type float.*/
    float exit2_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit2_x of type float.*/
    float exit2_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit2_y of type float.*/
    float exit3_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit3_x of type float.*/
    float exit3_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit3_y of type float.*/
    float exit4_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit4_x of type float.*/
    float exit4_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit4_y of type float.*/
    float exit5_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit5_x of type float.*/
    float exit5_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit5_y of type float.*/
    float exit6_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit6_x of type float.*/
    float exit6_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit6_y of type float.*/
    float exit7_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit7_x of type float.*/
    float exit7_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit7_y of type float.*/
    float exit8_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit8_x of type float.*/
    float exit8_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit8_y of type float.*/
    float exit9_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit9_x of type float.*/
    float exit9_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit9_y of type float.*/
    int drop_point [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list drop_point of type int.*/
    int sandbag_capacity [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list sandbag_capacity of type int.*/
    double nm_rough [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list nm_rough of type double.*/
    int evac_counter [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list evac_counter of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_WetDryMessage_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_WetDryMessage_list
{
    int inDomain [xmachine_message_WetDryMessage_MAX];    /**< Message memory variable list inDomain of type int.*/
    int x [xmachine_message_WetDryMessage_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_WetDryMessage_MAX];    /**< Message memory variable list y of type int.*/
    double min_hloc [xmachine_message_WetDryMessage_MAX];    /**< Message memory variable list min_hloc of type double.*/
    
};

/** struct xmachine_message_SpaceOperatorMessage_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_SpaceOperatorMessage_list
{
    int inDomain [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list inDomain of type int.*/
    int x [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list y of type int.*/
    double hFace_E [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list hFace_E of type double.*/
    double etFace_E [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list etFace_E of type double.*/
    double qFace_X_E [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_X_E of type double.*/
    double qFace_Y_E [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_Y_E of type double.*/
    double hFace_W [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list hFace_W of type double.*/
    double etFace_W [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list etFace_W of type double.*/
    double qFace_X_W [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_X_W of type double.*/
    double qFace_Y_W [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_Y_W of type double.*/
    double hFace_N [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list hFace_N of type double.*/
    double etFace_N [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list etFace_N of type double.*/
    double qFace_X_N [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_X_N of type double.*/
    double qFace_Y_N [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_Y_N of type double.*/
    double hFace_S [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list hFace_S of type double.*/
    double etFace_S [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list etFace_S of type double.*/
    double qFace_X_S [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_X_S of type double.*/
    double qFace_Y_S [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_Y_S of type double.*/
    
};

/** struct xmachine_message_FloodData_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_FloodData_list
{
    int inDomain [xmachine_message_FloodData_MAX];    /**< Message memory variable list inDomain of type int.*/
    int x [xmachine_message_FloodData_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_FloodData_MAX];    /**< Message memory variable list y of type int.*/
    double z0 [xmachine_message_FloodData_MAX];    /**< Message memory variable list z0 of type double.*/
    double h [xmachine_message_FloodData_MAX];    /**< Message memory variable list h of type double.*/
    double qx [xmachine_message_FloodData_MAX];    /**< Message memory variable list qx of type double.*/
    double qy [xmachine_message_FloodData_MAX];    /**< Message memory variable list qy of type double.*/
    double nm_rough [xmachine_message_FloodData_MAX];    /**< Message memory variable list nm_rough of type double.*/
    
};

/** struct xmachine_message_pedestrian_location_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_pedestrian_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_pedestrian_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_pedestrian_location_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list z of type float.*/
    
};

/** struct xmachine_message_PedData_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_PedData_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_PedData_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_PedData_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_PedData_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_PedData_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_PedData_MAX];    /**< Message memory variable list z of type float.*/
    int hero_status [xmachine_message_PedData_MAX];    /**< Message memory variable list hero_status of type int.*/
    int pickup_time [xmachine_message_PedData_MAX];    /**< Message memory variable list pickup_time of type int.*/
    int drop_time [xmachine_message_PedData_MAX];    /**< Message memory variable list drop_time of type int.*/
    int exit_no [xmachine_message_PedData_MAX];    /**< Message memory variable list exit_no of type int.*/
    int carry_sandbag [xmachine_message_PedData_MAX];    /**< Message memory variable list carry_sandbag of type int.*/
    double body_height [xmachine_message_PedData_MAX];    /**< Message memory variable list body_height of type double.*/
    
};

/** struct xmachine_message_updatedNavmapData_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_updatedNavmapData_list
{
    int x [xmachine_message_updatedNavmapData_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_updatedNavmapData_MAX];    /**< Message memory variable list y of type int.*/
    double z0 [xmachine_message_updatedNavmapData_MAX];    /**< Message memory variable list z0 of type double.*/
    int drop_point [xmachine_message_updatedNavmapData_MAX];    /**< Message memory variable list drop_point of type int.*/
    int sandbag_capacity [xmachine_message_updatedNavmapData_MAX];    /**< Message memory variable list sandbag_capacity of type int.*/
    int exit_no [xmachine_message_updatedNavmapData_MAX];    /**< Message memory variable list exit_no of type int.*/
    
};

/** struct xmachine_message_NavmapData_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_NavmapData_list
{
    int x [xmachine_message_NavmapData_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_NavmapData_MAX];    /**< Message memory variable list y of type int.*/
    double z0 [xmachine_message_NavmapData_MAX];    /**< Message memory variable list z0 of type double.*/
    double nm_rough [xmachine_message_NavmapData_MAX];    /**< Message memory variable list nm_rough of type double.*/
    
};

/** struct xmachine_message_navmap_cell_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_navmap_cell_list
{
    int x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list y of type int.*/
    double z0 [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list z0 of type double.*/
    double h [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list h of type double.*/
    double qx [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list qx of type double.*/
    double qy [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list qy of type double.*/
    int exit_no [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit_no of type int.*/
    float height [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list height of type float.*/
    float collision_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list collision_x of type float.*/
    float collision_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list collision_y of type float.*/
    float exit0_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit0_x of type float.*/
    float exit0_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit0_y of type float.*/
    float exit1_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit1_x of type float.*/
    float exit1_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit1_y of type float.*/
    float exit2_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit2_x of type float.*/
    float exit2_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit2_y of type float.*/
    float exit3_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit3_x of type float.*/
    float exit3_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit3_y of type float.*/
    float exit4_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit4_x of type float.*/
    float exit4_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit4_y of type float.*/
    float exit5_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit5_x of type float.*/
    float exit5_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit5_y of type float.*/
    float exit6_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit6_x of type float.*/
    float exit6_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit6_y of type float.*/
    float exit7_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit7_x of type float.*/
    float exit7_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit7_y of type float.*/
    float exit8_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit8_x of type float.*/
    float exit8_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit8_y of type float.*/
    float exit9_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit9_x of type float.*/
    float exit9_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit9_y of type float.*/
    
};



/* Spatially Partitioned Message boundary Matrices */

/** struct xmachine_message_pedestrian_location_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_pedestrian_location 
 */
struct xmachine_message_pedestrian_location_PBM
{
	int start[xmachine_message_pedestrian_location_grid_size];
	int end_or_count[xmachine_message_pedestrian_location_grid_size];
};

/** struct xmachine_message_PedData_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_PedData 
 */
struct xmachine_message_PedData_PBM
{
	int start[xmachine_message_PedData_grid_size];
	int end_or_count[xmachine_message_PedData_grid_size];
};



/* Graph structures */


/* Graph Edge Partitioned message boundary structures */


/* Graph utility functions, usable in agent functions and implemented in FLAMEGPU_Kernels */


  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * PrepareWetDry FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param WetDryMessage_messages Pointer to output message list of type xmachine_message_WetDryMessage_list. Must be passed as an argument to the add_WetDryMessage_message function ??.
 */
__FLAME_GPU_FUNC__ int PrepareWetDry(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WetDryMessage_messages);

/**
 * ProcessWetDryMessage FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param WetDryMessage_messages  WetDryMessage_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_WetDryMessage_message and get_next_WetDryMessage_message functions.
 */
__FLAME_GPU_FUNC__ int ProcessWetDryMessage(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WetDryMessage_messages);

/**
 * PrepareSpaceOperator FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param SpaceOperatorMessage_messages Pointer to output message list of type xmachine_message_SpaceOperatorMessage_list. Must be passed as an argument to the add_SpaceOperatorMessage_message function ??.
 */
__FLAME_GPU_FUNC__ int PrepareSpaceOperator(xmachine_memory_FloodCell* agent, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages);

/**
 * ProcessSpaceOperatorMessage FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param SpaceOperatorMessage_messages  SpaceOperatorMessage_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_SpaceOperatorMessage_message and get_next_SpaceOperatorMessage_message functions.
 */
__FLAME_GPU_FUNC__ int ProcessSpaceOperatorMessage(xmachine_memory_FloodCell* agent, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages);

/**
 * outputFloodData FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param FloodData_messages Pointer to output message list of type xmachine_message_FloodData_list. Must be passed as an argument to the add_FloodData_message function ??.
 */
__FLAME_GPU_FUNC__ int outputFloodData(xmachine_memory_FloodCell* agent, xmachine_message_FloodData_list* FloodData_messages);

/**
 * UpdateFloodTopo FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param NavmapData_messages  NavmapData_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_NavmapData_message and get_next_NavmapData_message functions.
 */
__FLAME_GPU_FUNC__ int UpdateFloodTopo(xmachine_memory_FloodCell* agent, xmachine_message_NavmapData_list* NavmapData_messages);

/**
 * output_pedestrian_location FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages Pointer to output message list of type xmachine_message_pedestrian_location_list. Must be passed as an argument to the add_pedestrian_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_pedestrian_location(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages);

/**
 * output_PedData FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param PedData_messages Pointer to output message list of type xmachine_message_PedData_list. Must be passed as an argument to the add_PedData_message function ??.
 */
__FLAME_GPU_FUNC__ int output_PedData(xmachine_memory_agent* agent, xmachine_message_PedData_list* PedData_messages);

/**
 * avoid_pedestrians FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages  pedestrian_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pedestrian_location_message and get_next_pedestrian_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_pedestrian_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int avoid_pedestrians(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48);

/**
 * force_flow FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages  navmap_cell_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_navmap_cell_message and get_next_navmap_cell_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int force_flow(xmachine_memory_agent* agent, xmachine_message_navmap_cell_list* navmap_cell_messages, RNG_rand48* rand48);

/**
 * move FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_agent* agent);

/**
 * output_navmap_cells FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages Pointer to output message list of type xmachine_message_navmap_cell_list. Must be passed as an argument to the add_navmap_cell_message function ??.
 */
__FLAME_GPU_FUNC__ int output_navmap_cells(xmachine_memory_navmap* agent, xmachine_message_navmap_cell_list* navmap_cell_messages);

/**
 * generate_pedestrians FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param agent_agents Pointer to agent list of type xmachine_memory_agent_list. This must be passed as an argument to the add_agent_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int generate_pedestrians(xmachine_memory_navmap* agent, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48);

/**
 * updateNavmap FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param FloodData_messages  FloodData_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_FloodData_message and get_next_FloodData_message functions.
 */
__FLAME_GPU_FUNC__ int updateNavmap(xmachine_memory_navmap* agent, xmachine_message_FloodData_list* FloodData_messages);

/**
 * updateNavmapData FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param PedData_messages  PedData_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_PedData_message and get_next_PedData_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_PedData_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param updatedNavmapData_messages Pointer to output message list of type xmachine_message_updatedNavmapData_list. Must be passed as an argument to the add_updatedNavmapData_message function ??.
 */
__FLAME_GPU_FUNC__ int updateNavmapData(xmachine_memory_navmap* agent, xmachine_message_PedData_list* PedData_messages, xmachine_message_PedData_PBM* partition_matrix, xmachine_message_updatedNavmapData_list* updatedNavmapData_messages);

/**
 * updateNeighbourNavmap FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param updatedNavmapData_messages  updatedNavmapData_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_updatedNavmapData_message and get_next_updatedNavmapData_message functions.* @param NavmapData_messages Pointer to output message list of type xmachine_message_NavmapData_list. Must be passed as an argument to the add_NavmapData_message function ??.
 */
__FLAME_GPU_FUNC__ int updateNeighbourNavmap(xmachine_memory_navmap* agent, xmachine_message_updatedNavmapData_list* updatedNavmapData_messages, xmachine_message_NavmapData_list* NavmapData_messages);

  
/* Message Function Prototypes for Discrete Partitioned WetDryMessage message implemented in FLAMEGPU_Kernels */

/** add_WetDryMessage_message
 * Function for all types of message partitioning
 * Adds a new WetDryMessage agent to the xmachine_memory_WetDryMessage_list list using a linear mapping
 * @param agents	xmachine_memory_WetDryMessage_list agent list
 * @param inDomain	message variable of type int
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param min_hloc	message variable of type double
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_WetDryMessage_message(xmachine_message_WetDryMessage_list* WetDryMessage_messages, int inDomain, int x, int y, double min_hloc);
 
/** get_first_WetDryMessage_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param WetDryMessage_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_WetDryMessage * get_first_WetDryMessage_message(xmachine_message_WetDryMessage_list* WetDryMessage_messages, int agentx, int agent_y);

/** get_next_WetDryMessage_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param WetDryMessage_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_WetDryMessage * get_next_WetDryMessage_message(xmachine_message_WetDryMessage* current, xmachine_message_WetDryMessage_list* WetDryMessage_messages);

  
/* Message Function Prototypes for Discrete Partitioned SpaceOperatorMessage message implemented in FLAMEGPU_Kernels */

/** add_SpaceOperatorMessage_message
 * Function for all types of message partitioning
 * Adds a new SpaceOperatorMessage agent to the xmachine_memory_SpaceOperatorMessage_list list using a linear mapping
 * @param agents	xmachine_memory_SpaceOperatorMessage_list agent list
 * @param inDomain	message variable of type int
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param hFace_E	message variable of type double
 * @param etFace_E	message variable of type double
 * @param qFace_X_E	message variable of type double
 * @param qFace_Y_E	message variable of type double
 * @param hFace_W	message variable of type double
 * @param etFace_W	message variable of type double
 * @param qFace_X_W	message variable of type double
 * @param qFace_Y_W	message variable of type double
 * @param hFace_N	message variable of type double
 * @param etFace_N	message variable of type double
 * @param qFace_X_N	message variable of type double
 * @param qFace_Y_N	message variable of type double
 * @param hFace_S	message variable of type double
 * @param etFace_S	message variable of type double
 * @param qFace_X_S	message variable of type double
 * @param qFace_Y_S	message variable of type double
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_SpaceOperatorMessage_message(xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages, int inDomain, int x, int y, double hFace_E, double etFace_E, double qFace_X_E, double qFace_Y_E, double hFace_W, double etFace_W, double qFace_X_W, double qFace_Y_W, double hFace_N, double etFace_N, double qFace_X_N, double qFace_Y_N, double hFace_S, double etFace_S, double qFace_X_S, double qFace_Y_S);
 
/** get_first_SpaceOperatorMessage_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param SpaceOperatorMessage_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_SpaceOperatorMessage * get_first_SpaceOperatorMessage_message(xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages, int agentx, int agent_y);

/** get_next_SpaceOperatorMessage_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param SpaceOperatorMessage_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_SpaceOperatorMessage * get_next_SpaceOperatorMessage_message(xmachine_message_SpaceOperatorMessage* current, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages);

  
/* Message Function Prototypes for Discrete Partitioned FloodData message implemented in FLAMEGPU_Kernels */

/** add_FloodData_message
 * Function for all types of message partitioning
 * Adds a new FloodData agent to the xmachine_memory_FloodData_list list using a linear mapping
 * @param agents	xmachine_memory_FloodData_list agent list
 * @param inDomain	message variable of type int
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param z0	message variable of type double
 * @param h	message variable of type double
 * @param qx	message variable of type double
 * @param qy	message variable of type double
 * @param nm_rough	message variable of type double
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_FloodData_message(xmachine_message_FloodData_list* FloodData_messages, int inDomain, int x, int y, double z0, double h, double qx, double qy, double nm_rough);
 
/** get_first_FloodData_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param FloodData_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_FloodData * get_first_FloodData_message(xmachine_message_FloodData_list* FloodData_messages, int agentx, int agent_y);

/** get_next_FloodData_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param FloodData_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_FloodData * get_next_FloodData_message(xmachine_message_FloodData* current, xmachine_message_FloodData_list* FloodData_messages);

  
/* Message Function Prototypes for Spatially Partitioned pedestrian_location message implemented in FLAMEGPU_Kernels */

/** add_pedestrian_location_message
 * Function for all types of message partitioning
 * Adds a new pedestrian_location agent to the xmachine_memory_pedestrian_location_list list using a linear mapping
 * @param agents	xmachine_memory_pedestrian_location_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_pedestrian_location_message(xmachine_message_pedestrian_location_list* pedestrian_location_messages, float x, float y, float z);
 
/** get_first_pedestrian_location_message
 * Get first message function for spatially partitioned messages
 * @param pedestrian_location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_location * get_first_pedestrian_location_message(xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, float x, float y, float z);

/** get_next_pedestrian_location_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param pedestrian_location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_location * get_next_pedestrian_location_message(xmachine_message_pedestrian_location* current, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix);

  
/* Message Function Prototypes for Spatially Partitioned PedData message implemented in FLAMEGPU_Kernels */

/** add_PedData_message
 * Function for all types of message partitioning
 * Adds a new PedData agent to the xmachine_memory_PedData_list list using a linear mapping
 * @param agents	xmachine_memory_PedData_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param hero_status	message variable of type int
 * @param pickup_time	message variable of type int
 * @param drop_time	message variable of type int
 * @param exit_no	message variable of type int
 * @param carry_sandbag	message variable of type int
 * @param body_height	message variable of type double
 */
 
 __FLAME_GPU_FUNC__ void add_PedData_message(xmachine_message_PedData_list* PedData_messages, float x, float y, float z, int hero_status, int pickup_time, int drop_time, int exit_no, int carry_sandbag, double body_height);
 
/** get_first_PedData_message
 * Get first message function for spatially partitioned messages
 * @param PedData_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_PedData * get_first_PedData_message(xmachine_message_PedData_list* PedData_messages, xmachine_message_PedData_PBM* partition_matrix, float x, float y, float z);

/** get_next_PedData_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param PedData_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_PedData * get_next_PedData_message(xmachine_message_PedData* current, xmachine_message_PedData_list* PedData_messages, xmachine_message_PedData_PBM* partition_matrix);

  
/* Message Function Prototypes for Discrete Partitioned updatedNavmapData message implemented in FLAMEGPU_Kernels */

/** add_updatedNavmapData_message
 * Function for all types of message partitioning
 * Adds a new updatedNavmapData agent to the xmachine_memory_updatedNavmapData_list list using a linear mapping
 * @param agents	xmachine_memory_updatedNavmapData_list agent list
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param z0	message variable of type double
 * @param drop_point	message variable of type int
 * @param sandbag_capacity	message variable of type int
 * @param exit_no	message variable of type int
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_updatedNavmapData_message(xmachine_message_updatedNavmapData_list* updatedNavmapData_messages, int x, int y, double z0, int drop_point, int sandbag_capacity, int exit_no);
 
/** get_first_updatedNavmapData_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param updatedNavmapData_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_updatedNavmapData * get_first_updatedNavmapData_message(xmachine_message_updatedNavmapData_list* updatedNavmapData_messages, int agentx, int agent_y);

/** get_next_updatedNavmapData_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param updatedNavmapData_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_updatedNavmapData * get_next_updatedNavmapData_message(xmachine_message_updatedNavmapData* current, xmachine_message_updatedNavmapData_list* updatedNavmapData_messages);

  
/* Message Function Prototypes for Discrete Partitioned NavmapData message implemented in FLAMEGPU_Kernels */

/** add_NavmapData_message
 * Function for all types of message partitioning
 * Adds a new NavmapData agent to the xmachine_memory_NavmapData_list list using a linear mapping
 * @param agents	xmachine_memory_NavmapData_list agent list
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param z0	message variable of type double
 * @param nm_rough	message variable of type double
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_NavmapData_message(xmachine_message_NavmapData_list* NavmapData_messages, int x, int y, double z0, double nm_rough);
 
/** get_first_NavmapData_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param NavmapData_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_NavmapData * get_first_NavmapData_message(xmachine_message_NavmapData_list* NavmapData_messages, int agentx, int agent_y);

/** get_next_NavmapData_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param NavmapData_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_NavmapData * get_next_NavmapData_message(xmachine_message_NavmapData* current, xmachine_message_NavmapData_list* NavmapData_messages);

  
/* Message Function Prototypes for Discrete Partitioned navmap_cell message implemented in FLAMEGPU_Kernels */

/** add_navmap_cell_message
 * Function for all types of message partitioning
 * Adds a new navmap_cell agent to the xmachine_memory_navmap_cell_list list using a linear mapping
 * @param agents	xmachine_memory_navmap_cell_list agent list
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param z0	message variable of type double
 * @param h	message variable of type double
 * @param qx	message variable of type double
 * @param qy	message variable of type double
 * @param exit_no	message variable of type int
 * @param height	message variable of type float
 * @param collision_x	message variable of type float
 * @param collision_y	message variable of type float
 * @param exit0_x	message variable of type float
 * @param exit0_y	message variable of type float
 * @param exit1_x	message variable of type float
 * @param exit1_y	message variable of type float
 * @param exit2_x	message variable of type float
 * @param exit2_y	message variable of type float
 * @param exit3_x	message variable of type float
 * @param exit3_y	message variable of type float
 * @param exit4_x	message variable of type float
 * @param exit4_y	message variable of type float
 * @param exit5_x	message variable of type float
 * @param exit5_y	message variable of type float
 * @param exit6_x	message variable of type float
 * @param exit6_y	message variable of type float
 * @param exit7_x	message variable of type float
 * @param exit7_y	message variable of type float
 * @param exit8_x	message variable of type float
 * @param exit8_y	message variable of type float
 * @param exit9_x	message variable of type float
 * @param exit9_y	message variable of type float
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_navmap_cell_message(xmachine_message_navmap_cell_list* navmap_cell_messages, int x, int y, double z0, double h, double qx, double qy, int exit_no, float height, float collision_x, float collision_y, float exit0_x, float exit0_y, float exit1_x, float exit1_y, float exit2_x, float exit2_y, float exit3_x, float exit3_y, float exit4_x, float exit4_y, float exit5_x, float exit5_y, float exit6_x, float exit6_y, float exit7_x, float exit7_y, float exit8_x, float exit8_y, float exit9_x, float exit9_y);
 
/** get_first_navmap_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param navmap_cell_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_navmap_cell * get_first_navmap_cell_message(xmachine_message_navmap_cell_list* navmap_cell_messages, int agentx, int agent_y);

/** get_next_navmap_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param navmap_cell_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_navmap_cell * get_next_navmap_cell_message(xmachine_message_navmap_cell* current, xmachine_message_navmap_cell_list* navmap_cell_messages);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_agent_agent
 * Adds a new continuous valued agent agent to the xmachine_memory_agent_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_agent_list agent list
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param velx	agent agent variable of type float
 * @param vely	agent agent variable of type float
 * @param steer_x	agent agent variable of type float
 * @param steer_y	agent agent variable of type float
 * @param height	agent agent variable of type float
 * @param exit_no	agent agent variable of type int
 * @param speed	agent agent variable of type float
 * @param lod	agent agent variable of type int
 * @param animate	agent agent variable of type float
 * @param animate_dir	agent agent variable of type int
 * @param HR_state	agent agent variable of type int
 * @param hero_status	agent agent variable of type int
 * @param pickup_time	agent agent variable of type double
 * @param drop_time	agent agent variable of type double
 * @param carry_sandbag	agent agent variable of type int
 * @param HR	agent agent variable of type double
 * @param dt_ped	agent agent variable of type float
 * @param d_water	agent agent variable of type float
 * @param v_water	agent agent variable of type float
 * @param body_height	agent agent variable of type float
 * @param body_mass	agent agent variable of type float
 * @param gender	agent agent variable of type int
 * @param stability_state	agent agent variable of type int
 * @param motion_speed	agent agent variable of type float
 * @param age	agent agent variable of type int
 * @param excitement_speed	agent agent variable of type float
 * @param dir_times	agent agent variable of type int
 * @param rejected_exit1	agent agent variable of type int
 * @param rejected_exit2	agent agent variable of type int
 * @param rejected_exit3	agent agent variable of type int
 * @param rejected_exit4	agent agent variable of type int
 * @param rejected_exit5	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_agent_agent(xmachine_memory_agent_list* agents, float x, float y, float velx, float vely, float steer_x, float steer_y, float height, int exit_no, float speed, int lod, float animate, int animate_dir, int HR_state, int hero_status, double pickup_time, double drop_time, int carry_sandbag, double HR, float dt_ped, float d_water, float v_water, float body_height, float body_mass, int gender, int stability_state, float motion_speed, int age, float excitement_speed, int dir_times, int rejected_exit1, int rejected_exit2, int rejected_exit3, int rejected_exit4, int rejected_exit5);


/* Graph loading function prototypes implemented in io.cu */


  
/* Simulation function prototypes implemented in simulation.cu */
/** getIterationNumber
 *  Get the iteration number (host)
 */
extern unsigned int getIterationNumber();

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input        XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 * @param h_FloodCells Pointer to agent list on the host
 * @param d_FloodCells Pointer to agent list on the GPU device
 * @param h_xmachine_memory_FloodCell_count Pointer to agent counter
 * @param h_agents Pointer to agent list on the host
 * @param d_agents Pointer to agent list on the GPU device
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 * @param h_navmaps Pointer to agent list on the host
 * @param d_navmaps Pointer to agent list on the GPU device
 * @param h_xmachine_memory_navmap_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_FloodCell_list* h_FloodCells_Default, xmachine_memory_FloodCell_list* d_FloodCells_Default, int h_xmachine_memory_FloodCell_Default_count,xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count,xmachine_memory_navmap_list* h_navmaps_static, xmachine_memory_navmap_list* d_navmaps_static, int h_xmachine_memory_navmap_static_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_FloodCells Pointer to agent list on the host
 * @param h_xmachine_memory_FloodCell_count Pointer to agent counter
 * @param h_agents Pointer to agent list on the host
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 * @param h_navmaps Pointer to agent list on the host
 * @param h_xmachine_memory_navmap_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_FloodCell_list* h_FloodCells, int* h_xmachine_memory_FloodCell_count,xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count,xmachine_memory_navmap_list* h_navmaps, int* h_xmachine_memory_navmap_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_FloodCell_MAX_count
 * Gets the max agent count for the FloodCell agent type 
 * @return		the maximum FloodCell agent count
 */
extern int get_agent_FloodCell_MAX_count();



/** get_agent_FloodCell_Default_count
 * Gets the agent count for the FloodCell agent type in state Default
 * @return		the current FloodCell agent count in state Default
 */
extern int get_agent_FloodCell_Default_count();

/** reset_Default_count
 * Resets the agent count of the FloodCell in state Default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_FloodCell_Default_count();

/** get_device_FloodCell_Default_agents
 * Gets a pointer to xmachine_memory_FloodCell_list on the GPU device
 * @return		a xmachine_memory_FloodCell_list on the GPU device
 */
extern xmachine_memory_FloodCell_list* get_device_FloodCell_Default_agents();

/** get_host_FloodCell_Default_agents
 * Gets a pointer to xmachine_memory_FloodCell_list on the CPU host
 * @return		a xmachine_memory_FloodCell_list on the CPU host
 */
extern xmachine_memory_FloodCell_list* get_host_FloodCell_Default_agents();


/** get_FloodCell_population_width
 * Gets an int value representing the xmachine_memory_FloodCell population width.
 * @return		xmachine_memory_FloodCell population width
 */
extern int get_FloodCell_population_width();

    
/** get_agent_agent_MAX_count
 * Gets the max agent count for the agent agent type 
 * @return		the maximum agent agent count
 */
extern int get_agent_agent_MAX_count();



/** get_agent_agent_default_count
 * Gets the agent count for the agent agent type in state default
 * @return		the current agent agent count in state default
 */
extern int get_agent_agent_default_count();

/** reset_default_count
 * Resets the agent count of the agent in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_agent_default_count();

/** get_device_agent_default_agents
 * Gets a pointer to xmachine_memory_agent_list on the GPU device
 * @return		a xmachine_memory_agent_list on the GPU device
 */
extern xmachine_memory_agent_list* get_device_agent_default_agents();

/** get_host_agent_default_agents
 * Gets a pointer to xmachine_memory_agent_list on the CPU host
 * @return		a xmachine_memory_agent_list on the CPU host
 */
extern xmachine_memory_agent_list* get_host_agent_default_agents();


/** sort_agents_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_agent_list* agents));


    
/** get_agent_navmap_MAX_count
 * Gets the max agent count for the navmap agent type 
 * @return		the maximum navmap agent count
 */
extern int get_agent_navmap_MAX_count();



/** get_agent_navmap_static_count
 * Gets the agent count for the navmap agent type in state static
 * @return		the current navmap agent count in state static
 */
extern int get_agent_navmap_static_count();

/** reset_static_count
 * Resets the agent count of the navmap in state static to 0. This is useful for interacting with some visualisations.
 */
extern void reset_navmap_static_count();

/** get_device_navmap_static_agents
 * Gets a pointer to xmachine_memory_navmap_list on the GPU device
 * @return		a xmachine_memory_navmap_list on the GPU device
 */
extern xmachine_memory_navmap_list* get_device_navmap_static_agents();

/** get_host_navmap_static_agents
 * Gets a pointer to xmachine_memory_navmap_list on the CPU host
 * @return		a xmachine_memory_navmap_list on the CPU host
 */
extern xmachine_memory_navmap_list* get_host_navmap_static_agents();


/** get_navmap_population_width
 * Gets an int value representing the xmachine_memory_navmap population width.
 * @return		xmachine_memory_navmap population width
 */
extern int get_navmap_population_width();


/* Host based access of agent variables*/

/** int get_FloodCell_Default_variable_inDomain(unsigned int index)
 * Gets the value of the inDomain variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable inDomain
 */
__host__ int get_FloodCell_Default_variable_inDomain(unsigned int index);

/** int get_FloodCell_Default_variable_x(unsigned int index)
 * Gets the value of the x variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_FloodCell_Default_variable_x(unsigned int index);

/** int get_FloodCell_Default_variable_y(unsigned int index)
 * Gets the value of the y variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_FloodCell_Default_variable_y(unsigned int index);

/** double get_FloodCell_Default_variable_z0(unsigned int index)
 * Gets the value of the z0 variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z0
 */
__host__ double get_FloodCell_Default_variable_z0(unsigned int index);

/** double get_FloodCell_Default_variable_h(unsigned int index)
 * Gets the value of the h variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable h
 */
__host__ double get_FloodCell_Default_variable_h(unsigned int index);

/** double get_FloodCell_Default_variable_qx(unsigned int index)
 * Gets the value of the qx variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qx
 */
__host__ double get_FloodCell_Default_variable_qx(unsigned int index);

/** double get_FloodCell_Default_variable_qy(unsigned int index)
 * Gets the value of the qy variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qy
 */
__host__ double get_FloodCell_Default_variable_qy(unsigned int index);

/** double get_FloodCell_Default_variable_timeStep(unsigned int index)
 * Gets the value of the timeStep variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable timeStep
 */
__host__ double get_FloodCell_Default_variable_timeStep(unsigned int index);

/** double get_FloodCell_Default_variable_minh_loc(unsigned int index)
 * Gets the value of the minh_loc variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable minh_loc
 */
__host__ double get_FloodCell_Default_variable_minh_loc(unsigned int index);

/** double get_FloodCell_Default_variable_hFace_E(unsigned int index)
 * Gets the value of the hFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_E
 */
__host__ double get_FloodCell_Default_variable_hFace_E(unsigned int index);

/** double get_FloodCell_Default_variable_etFace_E(unsigned int index)
 * Gets the value of the etFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_E
 */
__host__ double get_FloodCell_Default_variable_etFace_E(unsigned int index);

/** double get_FloodCell_Default_variable_qxFace_E(unsigned int index)
 * Gets the value of the qxFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_E
 */
__host__ double get_FloodCell_Default_variable_qxFace_E(unsigned int index);

/** double get_FloodCell_Default_variable_qyFace_E(unsigned int index)
 * Gets the value of the qyFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_E
 */
__host__ double get_FloodCell_Default_variable_qyFace_E(unsigned int index);

/** double get_FloodCell_Default_variable_hFace_W(unsigned int index)
 * Gets the value of the hFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_W
 */
__host__ double get_FloodCell_Default_variable_hFace_W(unsigned int index);

/** double get_FloodCell_Default_variable_etFace_W(unsigned int index)
 * Gets the value of the etFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_W
 */
__host__ double get_FloodCell_Default_variable_etFace_W(unsigned int index);

/** double get_FloodCell_Default_variable_qxFace_W(unsigned int index)
 * Gets the value of the qxFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_W
 */
__host__ double get_FloodCell_Default_variable_qxFace_W(unsigned int index);

/** double get_FloodCell_Default_variable_qyFace_W(unsigned int index)
 * Gets the value of the qyFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_W
 */
__host__ double get_FloodCell_Default_variable_qyFace_W(unsigned int index);

/** double get_FloodCell_Default_variable_hFace_N(unsigned int index)
 * Gets the value of the hFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_N
 */
__host__ double get_FloodCell_Default_variable_hFace_N(unsigned int index);

/** double get_FloodCell_Default_variable_etFace_N(unsigned int index)
 * Gets the value of the etFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_N
 */
__host__ double get_FloodCell_Default_variable_etFace_N(unsigned int index);

/** double get_FloodCell_Default_variable_qxFace_N(unsigned int index)
 * Gets the value of the qxFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_N
 */
__host__ double get_FloodCell_Default_variable_qxFace_N(unsigned int index);

/** double get_FloodCell_Default_variable_qyFace_N(unsigned int index)
 * Gets the value of the qyFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_N
 */
__host__ double get_FloodCell_Default_variable_qyFace_N(unsigned int index);

/** double get_FloodCell_Default_variable_hFace_S(unsigned int index)
 * Gets the value of the hFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_S
 */
__host__ double get_FloodCell_Default_variable_hFace_S(unsigned int index);

/** double get_FloodCell_Default_variable_etFace_S(unsigned int index)
 * Gets the value of the etFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_S
 */
__host__ double get_FloodCell_Default_variable_etFace_S(unsigned int index);

/** double get_FloodCell_Default_variable_qxFace_S(unsigned int index)
 * Gets the value of the qxFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_S
 */
__host__ double get_FloodCell_Default_variable_qxFace_S(unsigned int index);

/** double get_FloodCell_Default_variable_qyFace_S(unsigned int index)
 * Gets the value of the qyFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_S
 */
__host__ double get_FloodCell_Default_variable_qyFace_S(unsigned int index);

/** double get_FloodCell_Default_variable_nm_rough(unsigned int index)
 * Gets the value of the nm_rough variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable nm_rough
 */
__host__ double get_FloodCell_Default_variable_nm_rough(unsigned int index);

/** float get_agent_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_agent_default_variable_x(unsigned int index);

/** float get_agent_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_agent_default_variable_y(unsigned int index);

/** float get_agent_default_variable_velx(unsigned int index)
 * Gets the value of the velx variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable velx
 */
__host__ float get_agent_default_variable_velx(unsigned int index);

/** float get_agent_default_variable_vely(unsigned int index)
 * Gets the value of the vely variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable vely
 */
__host__ float get_agent_default_variable_vely(unsigned int index);

/** float get_agent_default_variable_steer_x(unsigned int index)
 * Gets the value of the steer_x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_x
 */
__host__ float get_agent_default_variable_steer_x(unsigned int index);

/** float get_agent_default_variable_steer_y(unsigned int index)
 * Gets the value of the steer_y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_y
 */
__host__ float get_agent_default_variable_steer_y(unsigned int index);

/** float get_agent_default_variable_height(unsigned int index)
 * Gets the value of the height variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_agent_default_variable_height(unsigned int index);

/** int get_agent_default_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_agent_default_variable_exit_no(unsigned int index);

/** float get_agent_default_variable_speed(unsigned int index)
 * Gets the value of the speed variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable speed
 */
__host__ float get_agent_default_variable_speed(unsigned int index);

/** int get_agent_default_variable_lod(unsigned int index)
 * Gets the value of the lod variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lod
 */
__host__ int get_agent_default_variable_lod(unsigned int index);

/** float get_agent_default_variable_animate(unsigned int index)
 * Gets the value of the animate variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate
 */
__host__ float get_agent_default_variable_animate(unsigned int index);

/** int get_agent_default_variable_animate_dir(unsigned int index)
 * Gets the value of the animate_dir variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate_dir
 */
__host__ int get_agent_default_variable_animate_dir(unsigned int index);

/** int get_agent_default_variable_HR_state(unsigned int index)
 * Gets the value of the HR_state variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable HR_state
 */
__host__ int get_agent_default_variable_HR_state(unsigned int index);

/** int get_agent_default_variable_hero_status(unsigned int index)
 * Gets the value of the hero_status variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hero_status
 */
__host__ int get_agent_default_variable_hero_status(unsigned int index);

/** double get_agent_default_variable_pickup_time(unsigned int index)
 * Gets the value of the pickup_time variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable pickup_time
 */
__host__ double get_agent_default_variable_pickup_time(unsigned int index);

/** double get_agent_default_variable_drop_time(unsigned int index)
 * Gets the value of the drop_time variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable drop_time
 */
__host__ double get_agent_default_variable_drop_time(unsigned int index);

/** int get_agent_default_variable_carry_sandbag(unsigned int index)
 * Gets the value of the carry_sandbag variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable carry_sandbag
 */
__host__ int get_agent_default_variable_carry_sandbag(unsigned int index);

/** double get_agent_default_variable_HR(unsigned int index)
 * Gets the value of the HR variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable HR
 */
__host__ double get_agent_default_variable_HR(unsigned int index);

/** float get_agent_default_variable_dt_ped(unsigned int index)
 * Gets the value of the dt_ped variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable dt_ped
 */
__host__ float get_agent_default_variable_dt_ped(unsigned int index);

/** float get_agent_default_variable_d_water(unsigned int index)
 * Gets the value of the d_water variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable d_water
 */
__host__ float get_agent_default_variable_d_water(unsigned int index);

/** float get_agent_default_variable_v_water(unsigned int index)
 * Gets the value of the v_water variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable v_water
 */
__host__ float get_agent_default_variable_v_water(unsigned int index);

/** float get_agent_default_variable_body_height(unsigned int index)
 * Gets the value of the body_height variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable body_height
 */
__host__ float get_agent_default_variable_body_height(unsigned int index);

/** float get_agent_default_variable_body_mass(unsigned int index)
 * Gets the value of the body_mass variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable body_mass
 */
__host__ float get_agent_default_variable_body_mass(unsigned int index);

/** int get_agent_default_variable_gender(unsigned int index)
 * Gets the value of the gender variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable gender
 */
__host__ int get_agent_default_variable_gender(unsigned int index);

/** int get_agent_default_variable_stability_state(unsigned int index)
 * Gets the value of the stability_state variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable stability_state
 */
__host__ int get_agent_default_variable_stability_state(unsigned int index);

/** float get_agent_default_variable_motion_speed(unsigned int index)
 * Gets the value of the motion_speed variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable motion_speed
 */
__host__ float get_agent_default_variable_motion_speed(unsigned int index);

/** int get_agent_default_variable_age(unsigned int index)
 * Gets the value of the age variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable age
 */
__host__ int get_agent_default_variable_age(unsigned int index);

/** float get_agent_default_variable_excitement_speed(unsigned int index)
 * Gets the value of the excitement_speed variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable excitement_speed
 */
__host__ float get_agent_default_variable_excitement_speed(unsigned int index);

/** int get_agent_default_variable_dir_times(unsigned int index)
 * Gets the value of the dir_times variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable dir_times
 */
__host__ int get_agent_default_variable_dir_times(unsigned int index);

/** int get_agent_default_variable_rejected_exit1(unsigned int index)
 * Gets the value of the rejected_exit1 variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rejected_exit1
 */
__host__ int get_agent_default_variable_rejected_exit1(unsigned int index);

/** int get_agent_default_variable_rejected_exit2(unsigned int index)
 * Gets the value of the rejected_exit2 variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rejected_exit2
 */
__host__ int get_agent_default_variable_rejected_exit2(unsigned int index);

/** int get_agent_default_variable_rejected_exit3(unsigned int index)
 * Gets the value of the rejected_exit3 variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rejected_exit3
 */
__host__ int get_agent_default_variable_rejected_exit3(unsigned int index);

/** int get_agent_default_variable_rejected_exit4(unsigned int index)
 * Gets the value of the rejected_exit4 variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rejected_exit4
 */
__host__ int get_agent_default_variable_rejected_exit4(unsigned int index);

/** int get_agent_default_variable_rejected_exit5(unsigned int index)
 * Gets the value of the rejected_exit5 variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rejected_exit5
 */
__host__ int get_agent_default_variable_rejected_exit5(unsigned int index);

/** int get_navmap_static_variable_x(unsigned int index)
 * Gets the value of the x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_navmap_static_variable_x(unsigned int index);

/** int get_navmap_static_variable_y(unsigned int index)
 * Gets the value of the y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_navmap_static_variable_y(unsigned int index);

/** double get_navmap_static_variable_z0(unsigned int index)
 * Gets the value of the z0 variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z0
 */
__host__ double get_navmap_static_variable_z0(unsigned int index);

/** double get_navmap_static_variable_h(unsigned int index)
 * Gets the value of the h variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable h
 */
__host__ double get_navmap_static_variable_h(unsigned int index);

/** double get_navmap_static_variable_qx(unsigned int index)
 * Gets the value of the qx variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qx
 */
__host__ double get_navmap_static_variable_qx(unsigned int index);

/** double get_navmap_static_variable_qy(unsigned int index)
 * Gets the value of the qy variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qy
 */
__host__ double get_navmap_static_variable_qy(unsigned int index);

/** int get_navmap_static_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_navmap_static_variable_exit_no(unsigned int index);

/** float get_navmap_static_variable_height(unsigned int index)
 * Gets the value of the height variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_navmap_static_variable_height(unsigned int index);

/** float get_navmap_static_variable_collision_x(unsigned int index)
 * Gets the value of the collision_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_x
 */
__host__ float get_navmap_static_variable_collision_x(unsigned int index);

/** float get_navmap_static_variable_collision_y(unsigned int index)
 * Gets the value of the collision_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_y
 */
__host__ float get_navmap_static_variable_collision_y(unsigned int index);

/** float get_navmap_static_variable_exit0_x(unsigned int index)
 * Gets the value of the exit0_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_x
 */
__host__ float get_navmap_static_variable_exit0_x(unsigned int index);

/** float get_navmap_static_variable_exit0_y(unsigned int index)
 * Gets the value of the exit0_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_y
 */
__host__ float get_navmap_static_variable_exit0_y(unsigned int index);

/** float get_navmap_static_variable_exit1_x(unsigned int index)
 * Gets the value of the exit1_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_x
 */
__host__ float get_navmap_static_variable_exit1_x(unsigned int index);

/** float get_navmap_static_variable_exit1_y(unsigned int index)
 * Gets the value of the exit1_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_y
 */
__host__ float get_navmap_static_variable_exit1_y(unsigned int index);

/** float get_navmap_static_variable_exit2_x(unsigned int index)
 * Gets the value of the exit2_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_x
 */
__host__ float get_navmap_static_variable_exit2_x(unsigned int index);

/** float get_navmap_static_variable_exit2_y(unsigned int index)
 * Gets the value of the exit2_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_y
 */
__host__ float get_navmap_static_variable_exit2_y(unsigned int index);

/** float get_navmap_static_variable_exit3_x(unsigned int index)
 * Gets the value of the exit3_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_x
 */
__host__ float get_navmap_static_variable_exit3_x(unsigned int index);

/** float get_navmap_static_variable_exit3_y(unsigned int index)
 * Gets the value of the exit3_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_y
 */
__host__ float get_navmap_static_variable_exit3_y(unsigned int index);

/** float get_navmap_static_variable_exit4_x(unsigned int index)
 * Gets the value of the exit4_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_x
 */
__host__ float get_navmap_static_variable_exit4_x(unsigned int index);

/** float get_navmap_static_variable_exit4_y(unsigned int index)
 * Gets the value of the exit4_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_y
 */
__host__ float get_navmap_static_variable_exit4_y(unsigned int index);

/** float get_navmap_static_variable_exit5_x(unsigned int index)
 * Gets the value of the exit5_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_x
 */
__host__ float get_navmap_static_variable_exit5_x(unsigned int index);

/** float get_navmap_static_variable_exit5_y(unsigned int index)
 * Gets the value of the exit5_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_y
 */
__host__ float get_navmap_static_variable_exit5_y(unsigned int index);

/** float get_navmap_static_variable_exit6_x(unsigned int index)
 * Gets the value of the exit6_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_x
 */
__host__ float get_navmap_static_variable_exit6_x(unsigned int index);

/** float get_navmap_static_variable_exit6_y(unsigned int index)
 * Gets the value of the exit6_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_y
 */
__host__ float get_navmap_static_variable_exit6_y(unsigned int index);

/** float get_navmap_static_variable_exit7_x(unsigned int index)
 * Gets the value of the exit7_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit7_x
 */
__host__ float get_navmap_static_variable_exit7_x(unsigned int index);

/** float get_navmap_static_variable_exit7_y(unsigned int index)
 * Gets the value of the exit7_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit7_y
 */
__host__ float get_navmap_static_variable_exit7_y(unsigned int index);

/** float get_navmap_static_variable_exit8_x(unsigned int index)
 * Gets the value of the exit8_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit8_x
 */
__host__ float get_navmap_static_variable_exit8_x(unsigned int index);

/** float get_navmap_static_variable_exit8_y(unsigned int index)
 * Gets the value of the exit8_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit8_y
 */
__host__ float get_navmap_static_variable_exit8_y(unsigned int index);

/** float get_navmap_static_variable_exit9_x(unsigned int index)
 * Gets the value of the exit9_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit9_x
 */
__host__ float get_navmap_static_variable_exit9_x(unsigned int index);

/** float get_navmap_static_variable_exit9_y(unsigned int index)
 * Gets the value of the exit9_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit9_y
 */
__host__ float get_navmap_static_variable_exit9_y(unsigned int index);

/** int get_navmap_static_variable_drop_point(unsigned int index)
 * Gets the value of the drop_point variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable drop_point
 */
__host__ int get_navmap_static_variable_drop_point(unsigned int index);

/** int get_navmap_static_variable_sandbag_capacity(unsigned int index)
 * Gets the value of the sandbag_capacity variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable sandbag_capacity
 */
__host__ int get_navmap_static_variable_sandbag_capacity(unsigned int index);

/** double get_navmap_static_variable_nm_rough(unsigned int index)
 * Gets the value of the nm_rough variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable nm_rough
 */
__host__ double get_navmap_static_variable_nm_rough(unsigned int index);

/** int get_navmap_static_variable_evac_counter(unsigned int index)
 * Gets the value of the evac_counter variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable evac_counter
 */
__host__ int get_navmap_static_variable_evac_counter(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_FloodCell
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated FloodCell struct.
 */
xmachine_memory_FloodCell* h_allocate_agent_FloodCell();
/** h_free_agent_FloodCell
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_FloodCell(xmachine_memory_FloodCell** agent);
/** h_allocate_agent_FloodCell_array
 * Utility function to allocate an array of structs for  FloodCell agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_FloodCell** h_allocate_agent_FloodCell_array(unsigned int count);
/** h_free_agent_FloodCell_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_FloodCell_array(xmachine_memory_FloodCell*** agents, unsigned int count);


/** h_add_agent_FloodCell_Default
 * Host function to add a single agent of type FloodCell to the Default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_FloodCell_Default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_FloodCell_Default(xmachine_memory_FloodCell* agent);

/** h_add_agents_FloodCell_Default(
 * Host function to add multiple agents of type FloodCell to the Default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of FloodCell agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_FloodCell_Default(xmachine_memory_FloodCell** agents, unsigned int count);

/** h_allocate_agent_agent
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated agent struct.
 */
xmachine_memory_agent* h_allocate_agent_agent();
/** h_free_agent_agent
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_agent(xmachine_memory_agent** agent);
/** h_allocate_agent_agent_array
 * Utility function to allocate an array of structs for  agent agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_agent** h_allocate_agent_agent_array(unsigned int count);
/** h_free_agent_agent_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_agent_array(xmachine_memory_agent*** agents, unsigned int count);


/** h_add_agent_agent_default
 * Host function to add a single agent of type agent to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_agent_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_agent_default(xmachine_memory_agent* agent);

/** h_add_agents_agent_default(
 * Host function to add multiple agents of type agent to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of agent agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_agent_default(xmachine_memory_agent** agents, unsigned int count);

/** h_allocate_agent_navmap
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated navmap struct.
 */
xmachine_memory_navmap* h_allocate_agent_navmap();
/** h_free_agent_navmap
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_navmap(xmachine_memory_navmap** agent);
/** h_allocate_agent_navmap_array
 * Utility function to allocate an array of structs for  navmap agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_navmap** h_allocate_agent_navmap_array(unsigned int count);
/** h_free_agent_navmap_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_navmap_array(xmachine_memory_navmap*** agents, unsigned int count);


/** h_add_agent_navmap_static
 * Host function to add a single agent of type navmap to the static state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_navmap_static instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_navmap_static(xmachine_memory_navmap* agent);

/** h_add_agents_navmap_static(
 * Host function to add multiple agents of type navmap to the static state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of navmap agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_navmap_static(xmachine_memory_navmap** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** int reduce_FloodCell_Default_inDomain_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_FloodCell_Default_inDomain_variable();



/** int count_FloodCell_Default_inDomain_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_FloodCell_Default_inDomain_variable(int count_value);

/** int min_FloodCell_Default_inDomain_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_FloodCell_Default_inDomain_variable();
/** int max_FloodCell_Default_inDomain_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_FloodCell_Default_inDomain_variable();

/** int reduce_FloodCell_Default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_FloodCell_Default_x_variable();



/** int count_FloodCell_Default_x_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_FloodCell_Default_x_variable(int count_value);

/** int min_FloodCell_Default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_FloodCell_Default_x_variable();
/** int max_FloodCell_Default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_FloodCell_Default_x_variable();

/** int reduce_FloodCell_Default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_FloodCell_Default_y_variable();



/** int count_FloodCell_Default_y_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_FloodCell_Default_y_variable(int count_value);

/** int min_FloodCell_Default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_FloodCell_Default_y_variable();
/** int max_FloodCell_Default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_FloodCell_Default_y_variable();

/** double reduce_FloodCell_Default_z0_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_z0_variable();



/** double min_FloodCell_Default_z0_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_z0_variable();
/** double max_FloodCell_Default_z0_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_z0_variable();

/** double reduce_FloodCell_Default_h_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_h_variable();



/** double min_FloodCell_Default_h_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_h_variable();
/** double max_FloodCell_Default_h_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_h_variable();

/** double reduce_FloodCell_Default_qx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qx_variable();



/** double min_FloodCell_Default_qx_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qx_variable();
/** double max_FloodCell_Default_qx_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qx_variable();

/** double reduce_FloodCell_Default_qy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qy_variable();



/** double min_FloodCell_Default_qy_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qy_variable();
/** double max_FloodCell_Default_qy_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qy_variable();

/** double reduce_FloodCell_Default_timeStep_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_timeStep_variable();



/** double min_FloodCell_Default_timeStep_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_timeStep_variable();
/** double max_FloodCell_Default_timeStep_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_timeStep_variable();

/** double reduce_FloodCell_Default_minh_loc_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_minh_loc_variable();



/** double min_FloodCell_Default_minh_loc_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_minh_loc_variable();
/** double max_FloodCell_Default_minh_loc_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_minh_loc_variable();

/** double reduce_FloodCell_Default_hFace_E_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_hFace_E_variable();



/** double min_FloodCell_Default_hFace_E_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_hFace_E_variable();
/** double max_FloodCell_Default_hFace_E_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_hFace_E_variable();

/** double reduce_FloodCell_Default_etFace_E_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_etFace_E_variable();



/** double min_FloodCell_Default_etFace_E_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_etFace_E_variable();
/** double max_FloodCell_Default_etFace_E_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_etFace_E_variable();

/** double reduce_FloodCell_Default_qxFace_E_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qxFace_E_variable();



/** double min_FloodCell_Default_qxFace_E_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qxFace_E_variable();
/** double max_FloodCell_Default_qxFace_E_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qxFace_E_variable();

/** double reduce_FloodCell_Default_qyFace_E_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qyFace_E_variable();



/** double min_FloodCell_Default_qyFace_E_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qyFace_E_variable();
/** double max_FloodCell_Default_qyFace_E_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qyFace_E_variable();

/** double reduce_FloodCell_Default_hFace_W_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_hFace_W_variable();



/** double min_FloodCell_Default_hFace_W_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_hFace_W_variable();
/** double max_FloodCell_Default_hFace_W_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_hFace_W_variable();

/** double reduce_FloodCell_Default_etFace_W_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_etFace_W_variable();



/** double min_FloodCell_Default_etFace_W_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_etFace_W_variable();
/** double max_FloodCell_Default_etFace_W_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_etFace_W_variable();

/** double reduce_FloodCell_Default_qxFace_W_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qxFace_W_variable();



/** double min_FloodCell_Default_qxFace_W_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qxFace_W_variable();
/** double max_FloodCell_Default_qxFace_W_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qxFace_W_variable();

/** double reduce_FloodCell_Default_qyFace_W_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qyFace_W_variable();



/** double min_FloodCell_Default_qyFace_W_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qyFace_W_variable();
/** double max_FloodCell_Default_qyFace_W_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qyFace_W_variable();

/** double reduce_FloodCell_Default_hFace_N_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_hFace_N_variable();



/** double min_FloodCell_Default_hFace_N_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_hFace_N_variable();
/** double max_FloodCell_Default_hFace_N_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_hFace_N_variable();

/** double reduce_FloodCell_Default_etFace_N_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_etFace_N_variable();



/** double min_FloodCell_Default_etFace_N_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_etFace_N_variable();
/** double max_FloodCell_Default_etFace_N_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_etFace_N_variable();

/** double reduce_FloodCell_Default_qxFace_N_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qxFace_N_variable();



/** double min_FloodCell_Default_qxFace_N_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qxFace_N_variable();
/** double max_FloodCell_Default_qxFace_N_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qxFace_N_variable();

/** double reduce_FloodCell_Default_qyFace_N_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qyFace_N_variable();



/** double min_FloodCell_Default_qyFace_N_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qyFace_N_variable();
/** double max_FloodCell_Default_qyFace_N_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qyFace_N_variable();

/** double reduce_FloodCell_Default_hFace_S_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_hFace_S_variable();



/** double min_FloodCell_Default_hFace_S_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_hFace_S_variable();
/** double max_FloodCell_Default_hFace_S_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_hFace_S_variable();

/** double reduce_FloodCell_Default_etFace_S_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_etFace_S_variable();



/** double min_FloodCell_Default_etFace_S_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_etFace_S_variable();
/** double max_FloodCell_Default_etFace_S_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_etFace_S_variable();

/** double reduce_FloodCell_Default_qxFace_S_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qxFace_S_variable();



/** double min_FloodCell_Default_qxFace_S_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qxFace_S_variable();
/** double max_FloodCell_Default_qxFace_S_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qxFace_S_variable();

/** double reduce_FloodCell_Default_qyFace_S_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qyFace_S_variable();



/** double min_FloodCell_Default_qyFace_S_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qyFace_S_variable();
/** double max_FloodCell_Default_qyFace_S_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qyFace_S_variable();

/** double reduce_FloodCell_Default_nm_rough_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_nm_rough_variable();



/** double min_FloodCell_Default_nm_rough_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_nm_rough_variable();
/** double max_FloodCell_Default_nm_rough_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_nm_rough_variable();

/** float reduce_agent_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_x_variable();



/** float min_agent_default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_x_variable();
/** float max_agent_default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_x_variable();

/** float reduce_agent_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_y_variable();



/** float min_agent_default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_y_variable();
/** float max_agent_default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_y_variable();

/** float reduce_agent_default_velx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_velx_variable();



/** float min_agent_default_velx_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_velx_variable();
/** float max_agent_default_velx_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_velx_variable();

/** float reduce_agent_default_vely_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_vely_variable();



/** float min_agent_default_vely_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_vely_variable();
/** float max_agent_default_vely_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_vely_variable();

/** float reduce_agent_default_steer_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_steer_x_variable();



/** float min_agent_default_steer_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_steer_x_variable();
/** float max_agent_default_steer_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_steer_x_variable();

/** float reduce_agent_default_steer_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_steer_y_variable();



/** float min_agent_default_steer_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_steer_y_variable();
/** float max_agent_default_steer_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_steer_y_variable();

/** float reduce_agent_default_height_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_height_variable();



/** float min_agent_default_height_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_height_variable();
/** float max_agent_default_height_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_height_variable();

/** int reduce_agent_default_exit_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_exit_no_variable();



/** int count_agent_default_exit_no_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_exit_no_variable(int count_value);

/** int min_agent_default_exit_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_exit_no_variable();
/** int max_agent_default_exit_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_exit_no_variable();

/** float reduce_agent_default_speed_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_speed_variable();



/** float min_agent_default_speed_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_speed_variable();
/** float max_agent_default_speed_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_speed_variable();

/** int reduce_agent_default_lod_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_lod_variable();



/** int count_agent_default_lod_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_lod_variable(int count_value);

/** int min_agent_default_lod_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_lod_variable();
/** int max_agent_default_lod_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_lod_variable();

/** float reduce_agent_default_animate_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_animate_variable();



/** float min_agent_default_animate_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_animate_variable();
/** float max_agent_default_animate_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_animate_variable();

/** int reduce_agent_default_animate_dir_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_animate_dir_variable();



/** int count_agent_default_animate_dir_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_animate_dir_variable(int count_value);

/** int min_agent_default_animate_dir_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_animate_dir_variable();
/** int max_agent_default_animate_dir_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_animate_dir_variable();

/** int reduce_agent_default_HR_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_HR_state_variable();



/** int count_agent_default_HR_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_HR_state_variable(int count_value);

/** int min_agent_default_HR_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_HR_state_variable();
/** int max_agent_default_HR_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_HR_state_variable();

/** int reduce_agent_default_hero_status_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_hero_status_variable();



/** int count_agent_default_hero_status_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_hero_status_variable(int count_value);

/** int min_agent_default_hero_status_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_hero_status_variable();
/** int max_agent_default_hero_status_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_hero_status_variable();

/** double reduce_agent_default_pickup_time_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_agent_default_pickup_time_variable();



/** double min_agent_default_pickup_time_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_agent_default_pickup_time_variable();
/** double max_agent_default_pickup_time_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_agent_default_pickup_time_variable();

/** double reduce_agent_default_drop_time_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_agent_default_drop_time_variable();



/** double min_agent_default_drop_time_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_agent_default_drop_time_variable();
/** double max_agent_default_drop_time_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_agent_default_drop_time_variable();

/** int reduce_agent_default_carry_sandbag_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_carry_sandbag_variable();



/** int count_agent_default_carry_sandbag_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_carry_sandbag_variable(int count_value);

/** int min_agent_default_carry_sandbag_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_carry_sandbag_variable();
/** int max_agent_default_carry_sandbag_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_carry_sandbag_variable();

/** double reduce_agent_default_HR_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_agent_default_HR_variable();



/** double min_agent_default_HR_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_agent_default_HR_variable();
/** double max_agent_default_HR_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_agent_default_HR_variable();

/** float reduce_agent_default_dt_ped_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_dt_ped_variable();



/** float min_agent_default_dt_ped_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_dt_ped_variable();
/** float max_agent_default_dt_ped_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_dt_ped_variable();

/** float reduce_agent_default_d_water_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_d_water_variable();



/** float min_agent_default_d_water_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_d_water_variable();
/** float max_agent_default_d_water_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_d_water_variable();

/** float reduce_agent_default_v_water_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_v_water_variable();



/** float min_agent_default_v_water_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_v_water_variable();
/** float max_agent_default_v_water_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_v_water_variable();

/** float reduce_agent_default_body_height_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_body_height_variable();



/** float min_agent_default_body_height_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_body_height_variable();
/** float max_agent_default_body_height_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_body_height_variable();

/** float reduce_agent_default_body_mass_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_body_mass_variable();



/** float min_agent_default_body_mass_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_body_mass_variable();
/** float max_agent_default_body_mass_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_body_mass_variable();

/** int reduce_agent_default_gender_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_gender_variable();



/** int count_agent_default_gender_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_gender_variable(int count_value);

/** int min_agent_default_gender_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_gender_variable();
/** int max_agent_default_gender_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_gender_variable();

/** int reduce_agent_default_stability_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_stability_state_variable();



/** int count_agent_default_stability_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_stability_state_variable(int count_value);

/** int min_agent_default_stability_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_stability_state_variable();
/** int max_agent_default_stability_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_stability_state_variable();

/** float reduce_agent_default_motion_speed_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_motion_speed_variable();



/** float min_agent_default_motion_speed_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_motion_speed_variable();
/** float max_agent_default_motion_speed_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_motion_speed_variable();

/** int reduce_agent_default_age_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_age_variable();



/** int count_agent_default_age_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_age_variable(int count_value);

/** int min_agent_default_age_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_age_variable();
/** int max_agent_default_age_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_age_variable();

/** float reduce_agent_default_excitement_speed_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_excitement_speed_variable();



/** float min_agent_default_excitement_speed_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_excitement_speed_variable();
/** float max_agent_default_excitement_speed_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_excitement_speed_variable();

/** int reduce_agent_default_dir_times_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_dir_times_variable();



/** int count_agent_default_dir_times_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_dir_times_variable(int count_value);

/** int min_agent_default_dir_times_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_dir_times_variable();
/** int max_agent_default_dir_times_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_dir_times_variable();

/** int reduce_agent_default_rejected_exit1_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_rejected_exit1_variable();



/** int count_agent_default_rejected_exit1_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_rejected_exit1_variable(int count_value);

/** int min_agent_default_rejected_exit1_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_rejected_exit1_variable();
/** int max_agent_default_rejected_exit1_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_rejected_exit1_variable();

/** int reduce_agent_default_rejected_exit2_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_rejected_exit2_variable();



/** int count_agent_default_rejected_exit2_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_rejected_exit2_variable(int count_value);

/** int min_agent_default_rejected_exit2_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_rejected_exit2_variable();
/** int max_agent_default_rejected_exit2_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_rejected_exit2_variable();

/** int reduce_agent_default_rejected_exit3_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_rejected_exit3_variable();



/** int count_agent_default_rejected_exit3_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_rejected_exit3_variable(int count_value);

/** int min_agent_default_rejected_exit3_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_rejected_exit3_variable();
/** int max_agent_default_rejected_exit3_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_rejected_exit3_variable();

/** int reduce_agent_default_rejected_exit4_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_rejected_exit4_variable();



/** int count_agent_default_rejected_exit4_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_rejected_exit4_variable(int count_value);

/** int min_agent_default_rejected_exit4_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_rejected_exit4_variable();
/** int max_agent_default_rejected_exit4_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_rejected_exit4_variable();

/** int reduce_agent_default_rejected_exit5_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_rejected_exit5_variable();



/** int count_agent_default_rejected_exit5_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_rejected_exit5_variable(int count_value);

/** int min_agent_default_rejected_exit5_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_rejected_exit5_variable();
/** int max_agent_default_rejected_exit5_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_rejected_exit5_variable();

/** int reduce_navmap_static_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_x_variable();



/** int count_navmap_static_x_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_x_variable(int count_value);

/** int min_navmap_static_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_x_variable();
/** int max_navmap_static_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_x_variable();

/** int reduce_navmap_static_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_y_variable();



/** int count_navmap_static_y_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_y_variable(int count_value);

/** int min_navmap_static_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_y_variable();
/** int max_navmap_static_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_y_variable();

/** double reduce_navmap_static_z0_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_navmap_static_z0_variable();



/** double min_navmap_static_z0_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_navmap_static_z0_variable();
/** double max_navmap_static_z0_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_navmap_static_z0_variable();

/** double reduce_navmap_static_h_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_navmap_static_h_variable();



/** double min_navmap_static_h_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_navmap_static_h_variable();
/** double max_navmap_static_h_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_navmap_static_h_variable();

/** double reduce_navmap_static_qx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_navmap_static_qx_variable();



/** double min_navmap_static_qx_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_navmap_static_qx_variable();
/** double max_navmap_static_qx_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_navmap_static_qx_variable();

/** double reduce_navmap_static_qy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_navmap_static_qy_variable();



/** double min_navmap_static_qy_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_navmap_static_qy_variable();
/** double max_navmap_static_qy_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_navmap_static_qy_variable();

/** int reduce_navmap_static_exit_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_exit_no_variable();



/** int count_navmap_static_exit_no_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_exit_no_variable(int count_value);

/** int min_navmap_static_exit_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_exit_no_variable();
/** int max_navmap_static_exit_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_exit_no_variable();

/** float reduce_navmap_static_height_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_height_variable();



/** float min_navmap_static_height_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_height_variable();
/** float max_navmap_static_height_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_height_variable();

/** float reduce_navmap_static_collision_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_collision_x_variable();



/** float min_navmap_static_collision_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_collision_x_variable();
/** float max_navmap_static_collision_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_collision_x_variable();

/** float reduce_navmap_static_collision_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_collision_y_variable();



/** float min_navmap_static_collision_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_collision_y_variable();
/** float max_navmap_static_collision_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_collision_y_variable();

/** float reduce_navmap_static_exit0_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit0_x_variable();



/** float min_navmap_static_exit0_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit0_x_variable();
/** float max_navmap_static_exit0_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit0_x_variable();

/** float reduce_navmap_static_exit0_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit0_y_variable();



/** float min_navmap_static_exit0_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit0_y_variable();
/** float max_navmap_static_exit0_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit0_y_variable();

/** float reduce_navmap_static_exit1_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit1_x_variable();



/** float min_navmap_static_exit1_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit1_x_variable();
/** float max_navmap_static_exit1_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit1_x_variable();

/** float reduce_navmap_static_exit1_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit1_y_variable();



/** float min_navmap_static_exit1_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit1_y_variable();
/** float max_navmap_static_exit1_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit1_y_variable();

/** float reduce_navmap_static_exit2_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit2_x_variable();



/** float min_navmap_static_exit2_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit2_x_variable();
/** float max_navmap_static_exit2_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit2_x_variable();

/** float reduce_navmap_static_exit2_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit2_y_variable();



/** float min_navmap_static_exit2_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit2_y_variable();
/** float max_navmap_static_exit2_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit2_y_variable();

/** float reduce_navmap_static_exit3_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit3_x_variable();



/** float min_navmap_static_exit3_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit3_x_variable();
/** float max_navmap_static_exit3_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit3_x_variable();

/** float reduce_navmap_static_exit3_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit3_y_variable();



/** float min_navmap_static_exit3_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit3_y_variable();
/** float max_navmap_static_exit3_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit3_y_variable();

/** float reduce_navmap_static_exit4_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit4_x_variable();



/** float min_navmap_static_exit4_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit4_x_variable();
/** float max_navmap_static_exit4_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit4_x_variable();

/** float reduce_navmap_static_exit4_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit4_y_variable();



/** float min_navmap_static_exit4_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit4_y_variable();
/** float max_navmap_static_exit4_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit4_y_variable();

/** float reduce_navmap_static_exit5_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit5_x_variable();



/** float min_navmap_static_exit5_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit5_x_variable();
/** float max_navmap_static_exit5_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit5_x_variable();

/** float reduce_navmap_static_exit5_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit5_y_variable();



/** float min_navmap_static_exit5_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit5_y_variable();
/** float max_navmap_static_exit5_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit5_y_variable();

/** float reduce_navmap_static_exit6_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit6_x_variable();



/** float min_navmap_static_exit6_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit6_x_variable();
/** float max_navmap_static_exit6_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit6_x_variable();

/** float reduce_navmap_static_exit6_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit6_y_variable();



/** float min_navmap_static_exit6_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit6_y_variable();
/** float max_navmap_static_exit6_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit6_y_variable();

/** float reduce_navmap_static_exit7_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit7_x_variable();



/** float min_navmap_static_exit7_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit7_x_variable();
/** float max_navmap_static_exit7_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit7_x_variable();

/** float reduce_navmap_static_exit7_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit7_y_variable();



/** float min_navmap_static_exit7_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit7_y_variable();
/** float max_navmap_static_exit7_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit7_y_variable();

/** float reduce_navmap_static_exit8_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit8_x_variable();



/** float min_navmap_static_exit8_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit8_x_variable();
/** float max_navmap_static_exit8_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit8_x_variable();

/** float reduce_navmap_static_exit8_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit8_y_variable();



/** float min_navmap_static_exit8_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit8_y_variable();
/** float max_navmap_static_exit8_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit8_y_variable();

/** float reduce_navmap_static_exit9_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit9_x_variable();



/** float min_navmap_static_exit9_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit9_x_variable();
/** float max_navmap_static_exit9_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit9_x_variable();

/** float reduce_navmap_static_exit9_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit9_y_variable();



/** float min_navmap_static_exit9_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit9_y_variable();
/** float max_navmap_static_exit9_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit9_y_variable();

/** int reduce_navmap_static_drop_point_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_drop_point_variable();



/** int count_navmap_static_drop_point_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_drop_point_variable(int count_value);

/** int min_navmap_static_drop_point_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_drop_point_variable();
/** int max_navmap_static_drop_point_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_drop_point_variable();

/** int reduce_navmap_static_sandbag_capacity_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_sandbag_capacity_variable();



/** int count_navmap_static_sandbag_capacity_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_sandbag_capacity_variable(int count_value);

/** int min_navmap_static_sandbag_capacity_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_sandbag_capacity_variable();
/** int max_navmap_static_sandbag_capacity_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_sandbag_capacity_variable();

/** double reduce_navmap_static_nm_rough_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_navmap_static_nm_rough_variable();



/** double min_navmap_static_nm_rough_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_navmap_static_nm_rough_variable();
/** double max_navmap_static_nm_rough_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_navmap_static_nm_rough_variable();

/** int reduce_navmap_static_evac_counter_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_evac_counter_variable();



/** int count_navmap_static_evac_counter_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_evac_counter_variable(int count_value);

/** int min_navmap_static_evac_counter_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_evac_counter_variable();
/** int max_navmap_static_evac_counter_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_evac_counter_variable();


  
/* global constant variables */

__constant__ double outputting_time;

__constant__ double outputting_time_interval;

__constant__ double xmin;

__constant__ double xmax;

__constant__ double ymin;

__constant__ double ymax;

__constant__ double dt_ped;

__constant__ double dt_flood;

__constant__ double dt;

__constant__ int auto_dt_on;

__constant__ int body_as_obstacle_on;

__constant__ int ped_roughness_effect_on;

__constant__ float body_height;

__constant__ float init_speed;

__constant__ float brisk_speed;

__constant__ double sim_time;

__constant__ double DXL;

__constant__ double DYL;

__constant__ double inflow_start_time;

__constant__ double inflow_peak_time;

__constant__ double inflow_end_time;

__constant__ double inflow_initial_discharge;

__constant__ double inflow_peak_discharge;

__constant__ double inflow_end_discharge;

__constant__ int INFLOW_BOUNDARY;

__constant__ int BOUNDARY_EAST_STATUS;

__constant__ int BOUNDARY_WEST_STATUS;

__constant__ int BOUNDARY_NORTH_STATUS;

__constant__ int BOUNDARY_SOUTH_STATUS;

__constant__ double x1_boundary;

__constant__ double x2_boundary;

__constant__ double y1_boundary;

__constant__ double y2_boundary;

__constant__ double init_depth_boundary;

__constant__ int evacuation_on;

__constant__ int walking_speed_reduction_in_water_on;

__constant__ int freeze_while_instable_on;

__constant__ double evacuation_end_time;

__constant__ double evacuation_start_time;

__constant__ int emergency_exit_number;

__constant__ int emer_alarm;

__constant__ double HR;

__constant__ int max_at_highest_risk;

__constant__ int max_at_low_risk;

__constant__ int max_at_medium_risk;

__constant__ int max_at_high_risk;

__constant__ double max_velocity;

__constant__ double max_depth;

__constant__ int count_population;

__constant__ int count_heros;

__constant__ int initial_population;

__constant__ int evacuated_population;

__constant__ float hero_percentage;

__constant__ int hero_population;

__constant__ int sandbagging_on;

__constant__ double sandbagging_start_time;

__constant__ double sandbagging_end_time;

__constant__ float sandbag_length;

__constant__ float sandbag_height;

__constant__ float sandbag_width;

__constant__ float extended_length;

__constant__ int sandbag_layers;

__constant__ int update_stopper;

__constant__ float dike_length;

__constant__ float dike_height;

__constant__ float dike_width;

__constant__ int fill_cap;

__constant__ int pickup_point;

__constant__ int drop_point;

__constant__ float pickup_duration;

__constant__ float drop_duration;

__constant__ float EMMISION_RATE_EXIT1;

__constant__ float EMMISION_RATE_EXIT2;

__constant__ float EMMISION_RATE_EXIT3;

__constant__ float EMMISION_RATE_EXIT4;

__constant__ float EMMISION_RATE_EXIT5;

__constant__ float EMMISION_RATE_EXIT6;

__constant__ float EMMISION_RATE_EXIT7;

__constant__ float EMMISION_RATE_EXIT8;

__constant__ float EMMISION_RATE_EXIT9;

__constant__ float EMMISION_RATE_EXIT10;

__constant__ int EXIT1_PROBABILITY;

__constant__ int EXIT2_PROBABILITY;

__constant__ int EXIT3_PROBABILITY;

__constant__ int EXIT4_PROBABILITY;

__constant__ int EXIT5_PROBABILITY;

__constant__ int EXIT6_PROBABILITY;

__constant__ int EXIT7_PROBABILITY;

__constant__ int EXIT8_PROBABILITY;

__constant__ int EXIT9_PROBABILITY;

__constant__ int EXIT10_PROBABILITY;

__constant__ int EXIT1_STATE;

__constant__ int EXIT2_STATE;

__constant__ int EXIT3_STATE;

__constant__ int EXIT4_STATE;

__constant__ int EXIT5_STATE;

__constant__ int EXIT6_STATE;

__constant__ int EXIT7_STATE;

__constant__ int EXIT8_STATE;

__constant__ int EXIT9_STATE;

__constant__ int EXIT10_STATE;

__constant__ int EXIT1_CELL_COUNT;

__constant__ int EXIT2_CELL_COUNT;

__constant__ int EXIT3_CELL_COUNT;

__constant__ int EXIT4_CELL_COUNT;

__constant__ int EXIT5_CELL_COUNT;

__constant__ int EXIT6_CELL_COUNT;

__constant__ int EXIT7_CELL_COUNT;

__constant__ int EXIT8_CELL_COUNT;

__constant__ int EXIT9_CELL_COUNT;

__constant__ int EXIT10_CELL_COUNT;

__constant__ float TIME_SCALER;

__constant__ float STEER_WEIGHT;

__constant__ float AVOID_WEIGHT;

__constant__ float COLLISION_WEIGHT;

__constant__ float GOAL_WEIGHT;

__constant__ int PedHeight_60_110_probability;

__constant__ int PedHeight_110_140_probability;

__constant__ int PedHeight_140_163_probability;

__constant__ int PedHeight_163_170_probability;

__constant__ int PedHeight_170_186_probability;

__constant__ int PedHeight_186_194_probability;

__constant__ int PedHeight_194_210_probability;

__constant__ int PedAge_10_17_probability;

__constant__ int PedAge_18_29_probability;

__constant__ int PedAge_30_39_probability;

__constant__ int PedAge_40_49_probability;

__constant__ int PedAge_50_59_probability;

__constant__ int PedAge_60_69_probability;

__constant__ int PedAge_70_79_probability;

__constant__ int excluded_age_probability;

__constant__ int gender_female_probability;

__constant__ int gender_male_probability;

__constant__ float SCALE_FACTOR;

__constant__ float I_SCALER;

__constant__ float MIN_DISTANCE;

__constant__ int excitement_on;

__constant__ int walk_run_switch;

__constant__ int preoccupying_on;

__constant__ int poly_hydrograph_on;

__constant__ int stop_emission_on;

__constant__ int goto_emergency_exit_on;

__constant__ int escape_route_finder_on;

__constant__ int dir_times;

__constant__ int no_return_on;

__constant__ float wdepth_perc_thresh;

__constant__ int follow_popular_exit_on;

__constant__ int popular_exit;

/** set_outputting_time
 * Sets the constant variable outputting_time on the device which can then be used in the agent functions.
 * @param h_outputting_time value to set the variable
 */
extern void set_outputting_time(double* h_outputting_time);

extern const double* get_outputting_time();


extern double h_env_outputting_time;

/** set_outputting_time_interval
 * Sets the constant variable outputting_time_interval on the device which can then be used in the agent functions.
 * @param h_outputting_time_interval value to set the variable
 */
extern void set_outputting_time_interval(double* h_outputting_time_interval);

extern const double* get_outputting_time_interval();


extern double h_env_outputting_time_interval;

/** set_xmin
 * Sets the constant variable xmin on the device which can then be used in the agent functions.
 * @param h_xmin value to set the variable
 */
extern void set_xmin(double* h_xmin);

extern const double* get_xmin();


extern double h_env_xmin;

/** set_xmax
 * Sets the constant variable xmax on the device which can then be used in the agent functions.
 * @param h_xmax value to set the variable
 */
extern void set_xmax(double* h_xmax);

extern const double* get_xmax();


extern double h_env_xmax;

/** set_ymin
 * Sets the constant variable ymin on the device which can then be used in the agent functions.
 * @param h_ymin value to set the variable
 */
extern void set_ymin(double* h_ymin);

extern const double* get_ymin();


extern double h_env_ymin;

/** set_ymax
 * Sets the constant variable ymax on the device which can then be used in the agent functions.
 * @param h_ymax value to set the variable
 */
extern void set_ymax(double* h_ymax);

extern const double* get_ymax();


extern double h_env_ymax;

/** set_dt_ped
 * Sets the constant variable dt_ped on the device which can then be used in the agent functions.
 * @param h_dt_ped value to set the variable
 */
extern void set_dt_ped(double* h_dt_ped);

extern const double* get_dt_ped();


extern double h_env_dt_ped;

/** set_dt_flood
 * Sets the constant variable dt_flood on the device which can then be used in the agent functions.
 * @param h_dt_flood value to set the variable
 */
extern void set_dt_flood(double* h_dt_flood);

extern const double* get_dt_flood();


extern double h_env_dt_flood;

/** set_dt
 * Sets the constant variable dt on the device which can then be used in the agent functions.
 * @param h_dt value to set the variable
 */
extern void set_dt(double* h_dt);

extern const double* get_dt();


extern double h_env_dt;

/** set_auto_dt_on
 * Sets the constant variable auto_dt_on on the device which can then be used in the agent functions.
 * @param h_auto_dt_on value to set the variable
 */
extern void set_auto_dt_on(int* h_auto_dt_on);

extern const int* get_auto_dt_on();


extern int h_env_auto_dt_on;

/** set_body_as_obstacle_on
 * Sets the constant variable body_as_obstacle_on on the device which can then be used in the agent functions.
 * @param h_body_as_obstacle_on value to set the variable
 */
extern void set_body_as_obstacle_on(int* h_body_as_obstacle_on);

extern const int* get_body_as_obstacle_on();


extern int h_env_body_as_obstacle_on;

/** set_ped_roughness_effect_on
 * Sets the constant variable ped_roughness_effect_on on the device which can then be used in the agent functions.
 * @param h_ped_roughness_effect_on value to set the variable
 */
extern void set_ped_roughness_effect_on(int* h_ped_roughness_effect_on);

extern const int* get_ped_roughness_effect_on();


extern int h_env_ped_roughness_effect_on;

/** set_body_height
 * Sets the constant variable body_height on the device which can then be used in the agent functions.
 * @param h_body_height value to set the variable
 */
extern void set_body_height(float* h_body_height);

extern const float* get_body_height();


extern float h_env_body_height;

/** set_init_speed
 * Sets the constant variable init_speed on the device which can then be used in the agent functions.
 * @param h_init_speed value to set the variable
 */
extern void set_init_speed(float* h_init_speed);

extern const float* get_init_speed();


extern float h_env_init_speed;

/** set_brisk_speed
 * Sets the constant variable brisk_speed on the device which can then be used in the agent functions.
 * @param h_brisk_speed value to set the variable
 */
extern void set_brisk_speed(float* h_brisk_speed);

extern const float* get_brisk_speed();


extern float h_env_brisk_speed;

/** set_sim_time
 * Sets the constant variable sim_time on the device which can then be used in the agent functions.
 * @param h_sim_time value to set the variable
 */
extern void set_sim_time(double* h_sim_time);

extern const double* get_sim_time();


extern double h_env_sim_time;

/** set_DXL
 * Sets the constant variable DXL on the device which can then be used in the agent functions.
 * @param h_DXL value to set the variable
 */
extern void set_DXL(double* h_DXL);

extern const double* get_DXL();


extern double h_env_DXL;

/** set_DYL
 * Sets the constant variable DYL on the device which can then be used in the agent functions.
 * @param h_DYL value to set the variable
 */
extern void set_DYL(double* h_DYL);

extern const double* get_DYL();


extern double h_env_DYL;

/** set_inflow_start_time
 * Sets the constant variable inflow_start_time on the device which can then be used in the agent functions.
 * @param h_inflow_start_time value to set the variable
 */
extern void set_inflow_start_time(double* h_inflow_start_time);

extern const double* get_inflow_start_time();


extern double h_env_inflow_start_time;

/** set_inflow_peak_time
 * Sets the constant variable inflow_peak_time on the device which can then be used in the agent functions.
 * @param h_inflow_peak_time value to set the variable
 */
extern void set_inflow_peak_time(double* h_inflow_peak_time);

extern const double* get_inflow_peak_time();


extern double h_env_inflow_peak_time;

/** set_inflow_end_time
 * Sets the constant variable inflow_end_time on the device which can then be used in the agent functions.
 * @param h_inflow_end_time value to set the variable
 */
extern void set_inflow_end_time(double* h_inflow_end_time);

extern const double* get_inflow_end_time();


extern double h_env_inflow_end_time;

/** set_inflow_initial_discharge
 * Sets the constant variable inflow_initial_discharge on the device which can then be used in the agent functions.
 * @param h_inflow_initial_discharge value to set the variable
 */
extern void set_inflow_initial_discharge(double* h_inflow_initial_discharge);

extern const double* get_inflow_initial_discharge();


extern double h_env_inflow_initial_discharge;

/** set_inflow_peak_discharge
 * Sets the constant variable inflow_peak_discharge on the device which can then be used in the agent functions.
 * @param h_inflow_peak_discharge value to set the variable
 */
extern void set_inflow_peak_discharge(double* h_inflow_peak_discharge);

extern const double* get_inflow_peak_discharge();


extern double h_env_inflow_peak_discharge;

/** set_inflow_end_discharge
 * Sets the constant variable inflow_end_discharge on the device which can then be used in the agent functions.
 * @param h_inflow_end_discharge value to set the variable
 */
extern void set_inflow_end_discharge(double* h_inflow_end_discharge);

extern const double* get_inflow_end_discharge();


extern double h_env_inflow_end_discharge;

/** set_INFLOW_BOUNDARY
 * Sets the constant variable INFLOW_BOUNDARY on the device which can then be used in the agent functions.
 * @param h_INFLOW_BOUNDARY value to set the variable
 */
extern void set_INFLOW_BOUNDARY(int* h_INFLOW_BOUNDARY);

extern const int* get_INFLOW_BOUNDARY();


extern int h_env_INFLOW_BOUNDARY;

/** set_BOUNDARY_EAST_STATUS
 * Sets the constant variable BOUNDARY_EAST_STATUS on the device which can then be used in the agent functions.
 * @param h_BOUNDARY_EAST_STATUS value to set the variable
 */
extern void set_BOUNDARY_EAST_STATUS(int* h_BOUNDARY_EAST_STATUS);

extern const int* get_BOUNDARY_EAST_STATUS();


extern int h_env_BOUNDARY_EAST_STATUS;

/** set_BOUNDARY_WEST_STATUS
 * Sets the constant variable BOUNDARY_WEST_STATUS on the device which can then be used in the agent functions.
 * @param h_BOUNDARY_WEST_STATUS value to set the variable
 */
extern void set_BOUNDARY_WEST_STATUS(int* h_BOUNDARY_WEST_STATUS);

extern const int* get_BOUNDARY_WEST_STATUS();


extern int h_env_BOUNDARY_WEST_STATUS;

/** set_BOUNDARY_NORTH_STATUS
 * Sets the constant variable BOUNDARY_NORTH_STATUS on the device which can then be used in the agent functions.
 * @param h_BOUNDARY_NORTH_STATUS value to set the variable
 */
extern void set_BOUNDARY_NORTH_STATUS(int* h_BOUNDARY_NORTH_STATUS);

extern const int* get_BOUNDARY_NORTH_STATUS();


extern int h_env_BOUNDARY_NORTH_STATUS;

/** set_BOUNDARY_SOUTH_STATUS
 * Sets the constant variable BOUNDARY_SOUTH_STATUS on the device which can then be used in the agent functions.
 * @param h_BOUNDARY_SOUTH_STATUS value to set the variable
 */
extern void set_BOUNDARY_SOUTH_STATUS(int* h_BOUNDARY_SOUTH_STATUS);

extern const int* get_BOUNDARY_SOUTH_STATUS();


extern int h_env_BOUNDARY_SOUTH_STATUS;

/** set_x1_boundary
 * Sets the constant variable x1_boundary on the device which can then be used in the agent functions.
 * @param h_x1_boundary value to set the variable
 */
extern void set_x1_boundary(double* h_x1_boundary);

extern const double* get_x1_boundary();


extern double h_env_x1_boundary;

/** set_x2_boundary
 * Sets the constant variable x2_boundary on the device which can then be used in the agent functions.
 * @param h_x2_boundary value to set the variable
 */
extern void set_x2_boundary(double* h_x2_boundary);

extern const double* get_x2_boundary();


extern double h_env_x2_boundary;

/** set_y1_boundary
 * Sets the constant variable y1_boundary on the device which can then be used in the agent functions.
 * @param h_y1_boundary value to set the variable
 */
extern void set_y1_boundary(double* h_y1_boundary);

extern const double* get_y1_boundary();


extern double h_env_y1_boundary;

/** set_y2_boundary
 * Sets the constant variable y2_boundary on the device which can then be used in the agent functions.
 * @param h_y2_boundary value to set the variable
 */
extern void set_y2_boundary(double* h_y2_boundary);

extern const double* get_y2_boundary();


extern double h_env_y2_boundary;

/** set_init_depth_boundary
 * Sets the constant variable init_depth_boundary on the device which can then be used in the agent functions.
 * @param h_init_depth_boundary value to set the variable
 */
extern void set_init_depth_boundary(double* h_init_depth_boundary);

extern const double* get_init_depth_boundary();


extern double h_env_init_depth_boundary;

/** set_evacuation_on
 * Sets the constant variable evacuation_on on the device which can then be used in the agent functions.
 * @param h_evacuation_on value to set the variable
 */
extern void set_evacuation_on(int* h_evacuation_on);

extern const int* get_evacuation_on();


extern int h_env_evacuation_on;

/** set_walking_speed_reduction_in_water_on
 * Sets the constant variable walking_speed_reduction_in_water_on on the device which can then be used in the agent functions.
 * @param h_walking_speed_reduction_in_water_on value to set the variable
 */
extern void set_walking_speed_reduction_in_water_on(int* h_walking_speed_reduction_in_water_on);

extern const int* get_walking_speed_reduction_in_water_on();


extern int h_env_walking_speed_reduction_in_water_on;

/** set_freeze_while_instable_on
 * Sets the constant variable freeze_while_instable_on on the device which can then be used in the agent functions.
 * @param h_freeze_while_instable_on value to set the variable
 */
extern void set_freeze_while_instable_on(int* h_freeze_while_instable_on);

extern const int* get_freeze_while_instable_on();


extern int h_env_freeze_while_instable_on;

/** set_evacuation_end_time
 * Sets the constant variable evacuation_end_time on the device which can then be used in the agent functions.
 * @param h_evacuation_end_time value to set the variable
 */
extern void set_evacuation_end_time(double* h_evacuation_end_time);

extern const double* get_evacuation_end_time();


extern double h_env_evacuation_end_time;

/** set_evacuation_start_time
 * Sets the constant variable evacuation_start_time on the device which can then be used in the agent functions.
 * @param h_evacuation_start_time value to set the variable
 */
extern void set_evacuation_start_time(double* h_evacuation_start_time);

extern const double* get_evacuation_start_time();


extern double h_env_evacuation_start_time;

/** set_emergency_exit_number
 * Sets the constant variable emergency_exit_number on the device which can then be used in the agent functions.
 * @param h_emergency_exit_number value to set the variable
 */
extern void set_emergency_exit_number(int* h_emergency_exit_number);

extern const int* get_emergency_exit_number();


extern int h_env_emergency_exit_number;

/** set_emer_alarm
 * Sets the constant variable emer_alarm on the device which can then be used in the agent functions.
 * @param h_emer_alarm value to set the variable
 */
extern void set_emer_alarm(int* h_emer_alarm);

extern const int* get_emer_alarm();


extern int h_env_emer_alarm;

/** set_HR
 * Sets the constant variable HR on the device which can then be used in the agent functions.
 * @param h_HR value to set the variable
 */
extern void set_HR(double* h_HR);

extern const double* get_HR();


extern double h_env_HR;

/** set_max_at_highest_risk
 * Sets the constant variable max_at_highest_risk on the device which can then be used in the agent functions.
 * @param h_max_at_highest_risk value to set the variable
 */
extern void set_max_at_highest_risk(int* h_max_at_highest_risk);

extern const int* get_max_at_highest_risk();


extern int h_env_max_at_highest_risk;

/** set_max_at_low_risk
 * Sets the constant variable max_at_low_risk on the device which can then be used in the agent functions.
 * @param h_max_at_low_risk value to set the variable
 */
extern void set_max_at_low_risk(int* h_max_at_low_risk);

extern const int* get_max_at_low_risk();


extern int h_env_max_at_low_risk;

/** set_max_at_medium_risk
 * Sets the constant variable max_at_medium_risk on the device which can then be used in the agent functions.
 * @param h_max_at_medium_risk value to set the variable
 */
extern void set_max_at_medium_risk(int* h_max_at_medium_risk);

extern const int* get_max_at_medium_risk();


extern int h_env_max_at_medium_risk;

/** set_max_at_high_risk
 * Sets the constant variable max_at_high_risk on the device which can then be used in the agent functions.
 * @param h_max_at_high_risk value to set the variable
 */
extern void set_max_at_high_risk(int* h_max_at_high_risk);

extern const int* get_max_at_high_risk();


extern int h_env_max_at_high_risk;

/** set_max_velocity
 * Sets the constant variable max_velocity on the device which can then be used in the agent functions.
 * @param h_max_velocity value to set the variable
 */
extern void set_max_velocity(double* h_max_velocity);

extern const double* get_max_velocity();


extern double h_env_max_velocity;

/** set_max_depth
 * Sets the constant variable max_depth on the device which can then be used in the agent functions.
 * @param h_max_depth value to set the variable
 */
extern void set_max_depth(double* h_max_depth);

extern const double* get_max_depth();


extern double h_env_max_depth;

/** set_count_population
 * Sets the constant variable count_population on the device which can then be used in the agent functions.
 * @param h_count_population value to set the variable
 */
extern void set_count_population(int* h_count_population);

extern const int* get_count_population();


extern int h_env_count_population;

/** set_count_heros
 * Sets the constant variable count_heros on the device which can then be used in the agent functions.
 * @param h_count_heros value to set the variable
 */
extern void set_count_heros(int* h_count_heros);

extern const int* get_count_heros();


extern int h_env_count_heros;

/** set_initial_population
 * Sets the constant variable initial_population on the device which can then be used in the agent functions.
 * @param h_initial_population value to set the variable
 */
extern void set_initial_population(int* h_initial_population);

extern const int* get_initial_population();


extern int h_env_initial_population;

/** set_evacuated_population
 * Sets the constant variable evacuated_population on the device which can then be used in the agent functions.
 * @param h_evacuated_population value to set the variable
 */
extern void set_evacuated_population(int* h_evacuated_population);

extern const int* get_evacuated_population();


extern int h_env_evacuated_population;

/** set_hero_percentage
 * Sets the constant variable hero_percentage on the device which can then be used in the agent functions.
 * @param h_hero_percentage value to set the variable
 */
extern void set_hero_percentage(float* h_hero_percentage);

extern const float* get_hero_percentage();


extern float h_env_hero_percentage;

/** set_hero_population
 * Sets the constant variable hero_population on the device which can then be used in the agent functions.
 * @param h_hero_population value to set the variable
 */
extern void set_hero_population(int* h_hero_population);

extern const int* get_hero_population();


extern int h_env_hero_population;

/** set_sandbagging_on
 * Sets the constant variable sandbagging_on on the device which can then be used in the agent functions.
 * @param h_sandbagging_on value to set the variable
 */
extern void set_sandbagging_on(int* h_sandbagging_on);

extern const int* get_sandbagging_on();


extern int h_env_sandbagging_on;

/** set_sandbagging_start_time
 * Sets the constant variable sandbagging_start_time on the device which can then be used in the agent functions.
 * @param h_sandbagging_start_time value to set the variable
 */
extern void set_sandbagging_start_time(double* h_sandbagging_start_time);

extern const double* get_sandbagging_start_time();


extern double h_env_sandbagging_start_time;

/** set_sandbagging_end_time
 * Sets the constant variable sandbagging_end_time on the device which can then be used in the agent functions.
 * @param h_sandbagging_end_time value to set the variable
 */
extern void set_sandbagging_end_time(double* h_sandbagging_end_time);

extern const double* get_sandbagging_end_time();


extern double h_env_sandbagging_end_time;

/** set_sandbag_length
 * Sets the constant variable sandbag_length on the device which can then be used in the agent functions.
 * @param h_sandbag_length value to set the variable
 */
extern void set_sandbag_length(float* h_sandbag_length);

extern const float* get_sandbag_length();


extern float h_env_sandbag_length;

/** set_sandbag_height
 * Sets the constant variable sandbag_height on the device which can then be used in the agent functions.
 * @param h_sandbag_height value to set the variable
 */
extern void set_sandbag_height(float* h_sandbag_height);

extern const float* get_sandbag_height();


extern float h_env_sandbag_height;

/** set_sandbag_width
 * Sets the constant variable sandbag_width on the device which can then be used in the agent functions.
 * @param h_sandbag_width value to set the variable
 */
extern void set_sandbag_width(float* h_sandbag_width);

extern const float* get_sandbag_width();


extern float h_env_sandbag_width;

/** set_extended_length
 * Sets the constant variable extended_length on the device which can then be used in the agent functions.
 * @param h_extended_length value to set the variable
 */
extern void set_extended_length(float* h_extended_length);

extern const float* get_extended_length();


extern float h_env_extended_length;

/** set_sandbag_layers
 * Sets the constant variable sandbag_layers on the device which can then be used in the agent functions.
 * @param h_sandbag_layers value to set the variable
 */
extern void set_sandbag_layers(int* h_sandbag_layers);

extern const int* get_sandbag_layers();


extern int h_env_sandbag_layers;

/** set_update_stopper
 * Sets the constant variable update_stopper on the device which can then be used in the agent functions.
 * @param h_update_stopper value to set the variable
 */
extern void set_update_stopper(int* h_update_stopper);

extern const int* get_update_stopper();


extern int h_env_update_stopper;

/** set_dike_length
 * Sets the constant variable dike_length on the device which can then be used in the agent functions.
 * @param h_dike_length value to set the variable
 */
extern void set_dike_length(float* h_dike_length);

extern const float* get_dike_length();


extern float h_env_dike_length;

/** set_dike_height
 * Sets the constant variable dike_height on the device which can then be used in the agent functions.
 * @param h_dike_height value to set the variable
 */
extern void set_dike_height(float* h_dike_height);

extern const float* get_dike_height();


extern float h_env_dike_height;

/** set_dike_width
 * Sets the constant variable dike_width on the device which can then be used in the agent functions.
 * @param h_dike_width value to set the variable
 */
extern void set_dike_width(float* h_dike_width);

extern const float* get_dike_width();


extern float h_env_dike_width;

/** set_fill_cap
 * Sets the constant variable fill_cap on the device which can then be used in the agent functions.
 * @param h_fill_cap value to set the variable
 */
extern void set_fill_cap(int* h_fill_cap);

extern const int* get_fill_cap();


extern int h_env_fill_cap;

/** set_pickup_point
 * Sets the constant variable pickup_point on the device which can then be used in the agent functions.
 * @param h_pickup_point value to set the variable
 */
extern void set_pickup_point(int* h_pickup_point);

extern const int* get_pickup_point();


extern int h_env_pickup_point;

/** set_drop_point
 * Sets the constant variable drop_point on the device which can then be used in the agent functions.
 * @param h_drop_point value to set the variable
 */
extern void set_drop_point(int* h_drop_point);

extern const int* get_drop_point();


extern int h_env_drop_point;

/** set_pickup_duration
 * Sets the constant variable pickup_duration on the device which can then be used in the agent functions.
 * @param h_pickup_duration value to set the variable
 */
extern void set_pickup_duration(float* h_pickup_duration);

extern const float* get_pickup_duration();


extern float h_env_pickup_duration;

/** set_drop_duration
 * Sets the constant variable drop_duration on the device which can then be used in the agent functions.
 * @param h_drop_duration value to set the variable
 */
extern void set_drop_duration(float* h_drop_duration);

extern const float* get_drop_duration();


extern float h_env_drop_duration;

/** set_EMMISION_RATE_EXIT1
 * Sets the constant variable EMMISION_RATE_EXIT1 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT1 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT1(float* h_EMMISION_RATE_EXIT1);

extern const float* get_EMMISION_RATE_EXIT1();


extern float h_env_EMMISION_RATE_EXIT1;

/** set_EMMISION_RATE_EXIT2
 * Sets the constant variable EMMISION_RATE_EXIT2 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT2 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT2(float* h_EMMISION_RATE_EXIT2);

extern const float* get_EMMISION_RATE_EXIT2();


extern float h_env_EMMISION_RATE_EXIT2;

/** set_EMMISION_RATE_EXIT3
 * Sets the constant variable EMMISION_RATE_EXIT3 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT3 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT3(float* h_EMMISION_RATE_EXIT3);

extern const float* get_EMMISION_RATE_EXIT3();


extern float h_env_EMMISION_RATE_EXIT3;

/** set_EMMISION_RATE_EXIT4
 * Sets the constant variable EMMISION_RATE_EXIT4 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT4 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT4(float* h_EMMISION_RATE_EXIT4);

extern const float* get_EMMISION_RATE_EXIT4();


extern float h_env_EMMISION_RATE_EXIT4;

/** set_EMMISION_RATE_EXIT5
 * Sets the constant variable EMMISION_RATE_EXIT5 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT5 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT5(float* h_EMMISION_RATE_EXIT5);

extern const float* get_EMMISION_RATE_EXIT5();


extern float h_env_EMMISION_RATE_EXIT5;

/** set_EMMISION_RATE_EXIT6
 * Sets the constant variable EMMISION_RATE_EXIT6 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT6 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT6(float* h_EMMISION_RATE_EXIT6);

extern const float* get_EMMISION_RATE_EXIT6();


extern float h_env_EMMISION_RATE_EXIT6;

/** set_EMMISION_RATE_EXIT7
 * Sets the constant variable EMMISION_RATE_EXIT7 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT7 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT7(float* h_EMMISION_RATE_EXIT7);

extern const float* get_EMMISION_RATE_EXIT7();


extern float h_env_EMMISION_RATE_EXIT7;

/** set_EMMISION_RATE_EXIT8
 * Sets the constant variable EMMISION_RATE_EXIT8 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT8 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT8(float* h_EMMISION_RATE_EXIT8);

extern const float* get_EMMISION_RATE_EXIT8();


extern float h_env_EMMISION_RATE_EXIT8;

/** set_EMMISION_RATE_EXIT9
 * Sets the constant variable EMMISION_RATE_EXIT9 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT9 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT9(float* h_EMMISION_RATE_EXIT9);

extern const float* get_EMMISION_RATE_EXIT9();


extern float h_env_EMMISION_RATE_EXIT9;

/** set_EMMISION_RATE_EXIT10
 * Sets the constant variable EMMISION_RATE_EXIT10 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT10 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT10(float* h_EMMISION_RATE_EXIT10);

extern const float* get_EMMISION_RATE_EXIT10();


extern float h_env_EMMISION_RATE_EXIT10;

/** set_EXIT1_PROBABILITY
 * Sets the constant variable EXIT1_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT1_PROBABILITY value to set the variable
 */
extern void set_EXIT1_PROBABILITY(int* h_EXIT1_PROBABILITY);

extern const int* get_EXIT1_PROBABILITY();


extern int h_env_EXIT1_PROBABILITY;

/** set_EXIT2_PROBABILITY
 * Sets the constant variable EXIT2_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT2_PROBABILITY value to set the variable
 */
extern void set_EXIT2_PROBABILITY(int* h_EXIT2_PROBABILITY);

extern const int* get_EXIT2_PROBABILITY();


extern int h_env_EXIT2_PROBABILITY;

/** set_EXIT3_PROBABILITY
 * Sets the constant variable EXIT3_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT3_PROBABILITY value to set the variable
 */
extern void set_EXIT3_PROBABILITY(int* h_EXIT3_PROBABILITY);

extern const int* get_EXIT3_PROBABILITY();


extern int h_env_EXIT3_PROBABILITY;

/** set_EXIT4_PROBABILITY
 * Sets the constant variable EXIT4_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT4_PROBABILITY value to set the variable
 */
extern void set_EXIT4_PROBABILITY(int* h_EXIT4_PROBABILITY);

extern const int* get_EXIT4_PROBABILITY();


extern int h_env_EXIT4_PROBABILITY;

/** set_EXIT5_PROBABILITY
 * Sets the constant variable EXIT5_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT5_PROBABILITY value to set the variable
 */
extern void set_EXIT5_PROBABILITY(int* h_EXIT5_PROBABILITY);

extern const int* get_EXIT5_PROBABILITY();


extern int h_env_EXIT5_PROBABILITY;

/** set_EXIT6_PROBABILITY
 * Sets the constant variable EXIT6_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT6_PROBABILITY value to set the variable
 */
extern void set_EXIT6_PROBABILITY(int* h_EXIT6_PROBABILITY);

extern const int* get_EXIT6_PROBABILITY();


extern int h_env_EXIT6_PROBABILITY;

/** set_EXIT7_PROBABILITY
 * Sets the constant variable EXIT7_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT7_PROBABILITY value to set the variable
 */
extern void set_EXIT7_PROBABILITY(int* h_EXIT7_PROBABILITY);

extern const int* get_EXIT7_PROBABILITY();


extern int h_env_EXIT7_PROBABILITY;

/** set_EXIT8_PROBABILITY
 * Sets the constant variable EXIT8_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT8_PROBABILITY value to set the variable
 */
extern void set_EXIT8_PROBABILITY(int* h_EXIT8_PROBABILITY);

extern const int* get_EXIT8_PROBABILITY();


extern int h_env_EXIT8_PROBABILITY;

/** set_EXIT9_PROBABILITY
 * Sets the constant variable EXIT9_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT9_PROBABILITY value to set the variable
 */
extern void set_EXIT9_PROBABILITY(int* h_EXIT9_PROBABILITY);

extern const int* get_EXIT9_PROBABILITY();


extern int h_env_EXIT9_PROBABILITY;

/** set_EXIT10_PROBABILITY
 * Sets the constant variable EXIT10_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT10_PROBABILITY value to set the variable
 */
extern void set_EXIT10_PROBABILITY(int* h_EXIT10_PROBABILITY);

extern const int* get_EXIT10_PROBABILITY();


extern int h_env_EXIT10_PROBABILITY;

/** set_EXIT1_STATE
 * Sets the constant variable EXIT1_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT1_STATE value to set the variable
 */
extern void set_EXIT1_STATE(int* h_EXIT1_STATE);

extern const int* get_EXIT1_STATE();


extern int h_env_EXIT1_STATE;

/** set_EXIT2_STATE
 * Sets the constant variable EXIT2_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT2_STATE value to set the variable
 */
extern void set_EXIT2_STATE(int* h_EXIT2_STATE);

extern const int* get_EXIT2_STATE();


extern int h_env_EXIT2_STATE;

/** set_EXIT3_STATE
 * Sets the constant variable EXIT3_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT3_STATE value to set the variable
 */
extern void set_EXIT3_STATE(int* h_EXIT3_STATE);

extern const int* get_EXIT3_STATE();


extern int h_env_EXIT3_STATE;

/** set_EXIT4_STATE
 * Sets the constant variable EXIT4_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT4_STATE value to set the variable
 */
extern void set_EXIT4_STATE(int* h_EXIT4_STATE);

extern const int* get_EXIT4_STATE();


extern int h_env_EXIT4_STATE;

/** set_EXIT5_STATE
 * Sets the constant variable EXIT5_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT5_STATE value to set the variable
 */
extern void set_EXIT5_STATE(int* h_EXIT5_STATE);

extern const int* get_EXIT5_STATE();


extern int h_env_EXIT5_STATE;

/** set_EXIT6_STATE
 * Sets the constant variable EXIT6_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT6_STATE value to set the variable
 */
extern void set_EXIT6_STATE(int* h_EXIT6_STATE);

extern const int* get_EXIT6_STATE();


extern int h_env_EXIT6_STATE;

/** set_EXIT7_STATE
 * Sets the constant variable EXIT7_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT7_STATE value to set the variable
 */
extern void set_EXIT7_STATE(int* h_EXIT7_STATE);

extern const int* get_EXIT7_STATE();


extern int h_env_EXIT7_STATE;

/** set_EXIT8_STATE
 * Sets the constant variable EXIT8_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT8_STATE value to set the variable
 */
extern void set_EXIT8_STATE(int* h_EXIT8_STATE);

extern const int* get_EXIT8_STATE();


extern int h_env_EXIT8_STATE;

/** set_EXIT9_STATE
 * Sets the constant variable EXIT9_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT9_STATE value to set the variable
 */
extern void set_EXIT9_STATE(int* h_EXIT9_STATE);

extern const int* get_EXIT9_STATE();


extern int h_env_EXIT9_STATE;

/** set_EXIT10_STATE
 * Sets the constant variable EXIT10_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT10_STATE value to set the variable
 */
extern void set_EXIT10_STATE(int* h_EXIT10_STATE);

extern const int* get_EXIT10_STATE();


extern int h_env_EXIT10_STATE;

/** set_EXIT1_CELL_COUNT
 * Sets the constant variable EXIT1_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT1_CELL_COUNT value to set the variable
 */
extern void set_EXIT1_CELL_COUNT(int* h_EXIT1_CELL_COUNT);

extern const int* get_EXIT1_CELL_COUNT();


extern int h_env_EXIT1_CELL_COUNT;

/** set_EXIT2_CELL_COUNT
 * Sets the constant variable EXIT2_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT2_CELL_COUNT value to set the variable
 */
extern void set_EXIT2_CELL_COUNT(int* h_EXIT2_CELL_COUNT);

extern const int* get_EXIT2_CELL_COUNT();


extern int h_env_EXIT2_CELL_COUNT;

/** set_EXIT3_CELL_COUNT
 * Sets the constant variable EXIT3_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT3_CELL_COUNT value to set the variable
 */
extern void set_EXIT3_CELL_COUNT(int* h_EXIT3_CELL_COUNT);

extern const int* get_EXIT3_CELL_COUNT();


extern int h_env_EXIT3_CELL_COUNT;

/** set_EXIT4_CELL_COUNT
 * Sets the constant variable EXIT4_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT4_CELL_COUNT value to set the variable
 */
extern void set_EXIT4_CELL_COUNT(int* h_EXIT4_CELL_COUNT);

extern const int* get_EXIT4_CELL_COUNT();


extern int h_env_EXIT4_CELL_COUNT;

/** set_EXIT5_CELL_COUNT
 * Sets the constant variable EXIT5_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT5_CELL_COUNT value to set the variable
 */
extern void set_EXIT5_CELL_COUNT(int* h_EXIT5_CELL_COUNT);

extern const int* get_EXIT5_CELL_COUNT();


extern int h_env_EXIT5_CELL_COUNT;

/** set_EXIT6_CELL_COUNT
 * Sets the constant variable EXIT6_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT6_CELL_COUNT value to set the variable
 */
extern void set_EXIT6_CELL_COUNT(int* h_EXIT6_CELL_COUNT);

extern const int* get_EXIT6_CELL_COUNT();


extern int h_env_EXIT6_CELL_COUNT;

/** set_EXIT7_CELL_COUNT
 * Sets the constant variable EXIT7_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT7_CELL_COUNT value to set the variable
 */
extern void set_EXIT7_CELL_COUNT(int* h_EXIT7_CELL_COUNT);

extern const int* get_EXIT7_CELL_COUNT();


extern int h_env_EXIT7_CELL_COUNT;

/** set_EXIT8_CELL_COUNT
 * Sets the constant variable EXIT8_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT8_CELL_COUNT value to set the variable
 */
extern void set_EXIT8_CELL_COUNT(int* h_EXIT8_CELL_COUNT);

extern const int* get_EXIT8_CELL_COUNT();


extern int h_env_EXIT8_CELL_COUNT;

/** set_EXIT9_CELL_COUNT
 * Sets the constant variable EXIT9_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT9_CELL_COUNT value to set the variable
 */
extern void set_EXIT9_CELL_COUNT(int* h_EXIT9_CELL_COUNT);

extern const int* get_EXIT9_CELL_COUNT();


extern int h_env_EXIT9_CELL_COUNT;

/** set_EXIT10_CELL_COUNT
 * Sets the constant variable EXIT10_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT10_CELL_COUNT value to set the variable
 */
extern void set_EXIT10_CELL_COUNT(int* h_EXIT10_CELL_COUNT);

extern const int* get_EXIT10_CELL_COUNT();


extern int h_env_EXIT10_CELL_COUNT;

/** set_TIME_SCALER
 * Sets the constant variable TIME_SCALER on the device which can then be used in the agent functions.
 * @param h_TIME_SCALER value to set the variable
 */
extern void set_TIME_SCALER(float* h_TIME_SCALER);

extern const float* get_TIME_SCALER();


extern float h_env_TIME_SCALER;

/** set_STEER_WEIGHT
 * Sets the constant variable STEER_WEIGHT on the device which can then be used in the agent functions.
 * @param h_STEER_WEIGHT value to set the variable
 */
extern void set_STEER_WEIGHT(float* h_STEER_WEIGHT);

extern const float* get_STEER_WEIGHT();


extern float h_env_STEER_WEIGHT;

/** set_AVOID_WEIGHT
 * Sets the constant variable AVOID_WEIGHT on the device which can then be used in the agent functions.
 * @param h_AVOID_WEIGHT value to set the variable
 */
extern void set_AVOID_WEIGHT(float* h_AVOID_WEIGHT);

extern const float* get_AVOID_WEIGHT();


extern float h_env_AVOID_WEIGHT;

/** set_COLLISION_WEIGHT
 * Sets the constant variable COLLISION_WEIGHT on the device which can then be used in the agent functions.
 * @param h_COLLISION_WEIGHT value to set the variable
 */
extern void set_COLLISION_WEIGHT(float* h_COLLISION_WEIGHT);

extern const float* get_COLLISION_WEIGHT();


extern float h_env_COLLISION_WEIGHT;

/** set_GOAL_WEIGHT
 * Sets the constant variable GOAL_WEIGHT on the device which can then be used in the agent functions.
 * @param h_GOAL_WEIGHT value to set the variable
 */
extern void set_GOAL_WEIGHT(float* h_GOAL_WEIGHT);

extern const float* get_GOAL_WEIGHT();


extern float h_env_GOAL_WEIGHT;

/** set_PedHeight_60_110_probability
 * Sets the constant variable PedHeight_60_110_probability on the device which can then be used in the agent functions.
 * @param h_PedHeight_60_110_probability value to set the variable
 */
extern void set_PedHeight_60_110_probability(int* h_PedHeight_60_110_probability);

extern const int* get_PedHeight_60_110_probability();


extern int h_env_PedHeight_60_110_probability;

/** set_PedHeight_110_140_probability
 * Sets the constant variable PedHeight_110_140_probability on the device which can then be used in the agent functions.
 * @param h_PedHeight_110_140_probability value to set the variable
 */
extern void set_PedHeight_110_140_probability(int* h_PedHeight_110_140_probability);

extern const int* get_PedHeight_110_140_probability();


extern int h_env_PedHeight_110_140_probability;

/** set_PedHeight_140_163_probability
 * Sets the constant variable PedHeight_140_163_probability on the device which can then be used in the agent functions.
 * @param h_PedHeight_140_163_probability value to set the variable
 */
extern void set_PedHeight_140_163_probability(int* h_PedHeight_140_163_probability);

extern const int* get_PedHeight_140_163_probability();


extern int h_env_PedHeight_140_163_probability;

/** set_PedHeight_163_170_probability
 * Sets the constant variable PedHeight_163_170_probability on the device which can then be used in the agent functions.
 * @param h_PedHeight_163_170_probability value to set the variable
 */
extern void set_PedHeight_163_170_probability(int* h_PedHeight_163_170_probability);

extern const int* get_PedHeight_163_170_probability();


extern int h_env_PedHeight_163_170_probability;

/** set_PedHeight_170_186_probability
 * Sets the constant variable PedHeight_170_186_probability on the device which can then be used in the agent functions.
 * @param h_PedHeight_170_186_probability value to set the variable
 */
extern void set_PedHeight_170_186_probability(int* h_PedHeight_170_186_probability);

extern const int* get_PedHeight_170_186_probability();


extern int h_env_PedHeight_170_186_probability;

/** set_PedHeight_186_194_probability
 * Sets the constant variable PedHeight_186_194_probability on the device which can then be used in the agent functions.
 * @param h_PedHeight_186_194_probability value to set the variable
 */
extern void set_PedHeight_186_194_probability(int* h_PedHeight_186_194_probability);

extern const int* get_PedHeight_186_194_probability();


extern int h_env_PedHeight_186_194_probability;

/** set_PedHeight_194_210_probability
 * Sets the constant variable PedHeight_194_210_probability on the device which can then be used in the agent functions.
 * @param h_PedHeight_194_210_probability value to set the variable
 */
extern void set_PedHeight_194_210_probability(int* h_PedHeight_194_210_probability);

extern const int* get_PedHeight_194_210_probability();


extern int h_env_PedHeight_194_210_probability;

/** set_PedAge_10_17_probability
 * Sets the constant variable PedAge_10_17_probability on the device which can then be used in the agent functions.
 * @param h_PedAge_10_17_probability value to set the variable
 */
extern void set_PedAge_10_17_probability(int* h_PedAge_10_17_probability);

extern const int* get_PedAge_10_17_probability();


extern int h_env_PedAge_10_17_probability;

/** set_PedAge_18_29_probability
 * Sets the constant variable PedAge_18_29_probability on the device which can then be used in the agent functions.
 * @param h_PedAge_18_29_probability value to set the variable
 */
extern void set_PedAge_18_29_probability(int* h_PedAge_18_29_probability);

extern const int* get_PedAge_18_29_probability();


extern int h_env_PedAge_18_29_probability;

/** set_PedAge_30_39_probability
 * Sets the constant variable PedAge_30_39_probability on the device which can then be used in the agent functions.
 * @param h_PedAge_30_39_probability value to set the variable
 */
extern void set_PedAge_30_39_probability(int* h_PedAge_30_39_probability);

extern const int* get_PedAge_30_39_probability();


extern int h_env_PedAge_30_39_probability;

/** set_PedAge_40_49_probability
 * Sets the constant variable PedAge_40_49_probability on the device which can then be used in the agent functions.
 * @param h_PedAge_40_49_probability value to set the variable
 */
extern void set_PedAge_40_49_probability(int* h_PedAge_40_49_probability);

extern const int* get_PedAge_40_49_probability();


extern int h_env_PedAge_40_49_probability;

/** set_PedAge_50_59_probability
 * Sets the constant variable PedAge_50_59_probability on the device which can then be used in the agent functions.
 * @param h_PedAge_50_59_probability value to set the variable
 */
extern void set_PedAge_50_59_probability(int* h_PedAge_50_59_probability);

extern const int* get_PedAge_50_59_probability();


extern int h_env_PedAge_50_59_probability;

/** set_PedAge_60_69_probability
 * Sets the constant variable PedAge_60_69_probability on the device which can then be used in the agent functions.
 * @param h_PedAge_60_69_probability value to set the variable
 */
extern void set_PedAge_60_69_probability(int* h_PedAge_60_69_probability);

extern const int* get_PedAge_60_69_probability();


extern int h_env_PedAge_60_69_probability;

/** set_PedAge_70_79_probability
 * Sets the constant variable PedAge_70_79_probability on the device which can then be used in the agent functions.
 * @param h_PedAge_70_79_probability value to set the variable
 */
extern void set_PedAge_70_79_probability(int* h_PedAge_70_79_probability);

extern const int* get_PedAge_70_79_probability();


extern int h_env_PedAge_70_79_probability;

/** set_excluded_age_probability
 * Sets the constant variable excluded_age_probability on the device which can then be used in the agent functions.
 * @param h_excluded_age_probability value to set the variable
 */
extern void set_excluded_age_probability(int* h_excluded_age_probability);

extern const int* get_excluded_age_probability();


extern int h_env_excluded_age_probability;

/** set_gender_female_probability
 * Sets the constant variable gender_female_probability on the device which can then be used in the agent functions.
 * @param h_gender_female_probability value to set the variable
 */
extern void set_gender_female_probability(int* h_gender_female_probability);

extern const int* get_gender_female_probability();


extern int h_env_gender_female_probability;

/** set_gender_male_probability
 * Sets the constant variable gender_male_probability on the device which can then be used in the agent functions.
 * @param h_gender_male_probability value to set the variable
 */
extern void set_gender_male_probability(int* h_gender_male_probability);

extern const int* get_gender_male_probability();


extern int h_env_gender_male_probability;

/** set_SCALE_FACTOR
 * Sets the constant variable SCALE_FACTOR on the device which can then be used in the agent functions.
 * @param h_SCALE_FACTOR value to set the variable
 */
extern void set_SCALE_FACTOR(float* h_SCALE_FACTOR);

extern const float* get_SCALE_FACTOR();


extern float h_env_SCALE_FACTOR;

/** set_I_SCALER
 * Sets the constant variable I_SCALER on the device which can then be used in the agent functions.
 * @param h_I_SCALER value to set the variable
 */
extern void set_I_SCALER(float* h_I_SCALER);

extern const float* get_I_SCALER();


extern float h_env_I_SCALER;

/** set_MIN_DISTANCE
 * Sets the constant variable MIN_DISTANCE on the device which can then be used in the agent functions.
 * @param h_MIN_DISTANCE value to set the variable
 */
extern void set_MIN_DISTANCE(float* h_MIN_DISTANCE);

extern const float* get_MIN_DISTANCE();


extern float h_env_MIN_DISTANCE;

/** set_excitement_on
 * Sets the constant variable excitement_on on the device which can then be used in the agent functions.
 * @param h_excitement_on value to set the variable
 */
extern void set_excitement_on(int* h_excitement_on);

extern const int* get_excitement_on();


extern int h_env_excitement_on;

/** set_walk_run_switch
 * Sets the constant variable walk_run_switch on the device which can then be used in the agent functions.
 * @param h_walk_run_switch value to set the variable
 */
extern void set_walk_run_switch(int* h_walk_run_switch);

extern const int* get_walk_run_switch();


extern int h_env_walk_run_switch;

/** set_preoccupying_on
 * Sets the constant variable preoccupying_on on the device which can then be used in the agent functions.
 * @param h_preoccupying_on value to set the variable
 */
extern void set_preoccupying_on(int* h_preoccupying_on);

extern const int* get_preoccupying_on();


extern int h_env_preoccupying_on;

/** set_poly_hydrograph_on
 * Sets the constant variable poly_hydrograph_on on the device which can then be used in the agent functions.
 * @param h_poly_hydrograph_on value to set the variable
 */
extern void set_poly_hydrograph_on(int* h_poly_hydrograph_on);

extern const int* get_poly_hydrograph_on();


extern int h_env_poly_hydrograph_on;

/** set_stop_emission_on
 * Sets the constant variable stop_emission_on on the device which can then be used in the agent functions.
 * @param h_stop_emission_on value to set the variable
 */
extern void set_stop_emission_on(int* h_stop_emission_on);

extern const int* get_stop_emission_on();


extern int h_env_stop_emission_on;

/** set_goto_emergency_exit_on
 * Sets the constant variable goto_emergency_exit_on on the device which can then be used in the agent functions.
 * @param h_goto_emergency_exit_on value to set the variable
 */
extern void set_goto_emergency_exit_on(int* h_goto_emergency_exit_on);

extern const int* get_goto_emergency_exit_on();


extern int h_env_goto_emergency_exit_on;

/** set_escape_route_finder_on
 * Sets the constant variable escape_route_finder_on on the device which can then be used in the agent functions.
 * @param h_escape_route_finder_on value to set the variable
 */
extern void set_escape_route_finder_on(int* h_escape_route_finder_on);

extern const int* get_escape_route_finder_on();


extern int h_env_escape_route_finder_on;

/** set_dir_times
 * Sets the constant variable dir_times on the device which can then be used in the agent functions.
 * @param h_dir_times value to set the variable
 */
extern void set_dir_times(int* h_dir_times);

extern const int* get_dir_times();


extern int h_env_dir_times;

/** set_no_return_on
 * Sets the constant variable no_return_on on the device which can then be used in the agent functions.
 * @param h_no_return_on value to set the variable
 */
extern void set_no_return_on(int* h_no_return_on);

extern const int* get_no_return_on();


extern int h_env_no_return_on;

/** set_wdepth_perc_thresh
 * Sets the constant variable wdepth_perc_thresh on the device which can then be used in the agent functions.
 * @param h_wdepth_perc_thresh value to set the variable
 */
extern void set_wdepth_perc_thresh(float* h_wdepth_perc_thresh);

extern const float* get_wdepth_perc_thresh();


extern float h_env_wdepth_perc_thresh;

/** set_follow_popular_exit_on
 * Sets the constant variable follow_popular_exit_on on the device which can then be used in the agent functions.
 * @param h_follow_popular_exit_on value to set the variable
 */
extern void set_follow_popular_exit_on(int* h_follow_popular_exit_on);

extern const int* get_follow_popular_exit_on();


extern int h_env_follow_popular_exit_on;

/** set_popular_exit
 * Sets the constant variable popular_exit on the device which can then be used in the agent functions.
 * @param h_popular_exit value to set the variable
 */
extern void set_popular_exit(int* h_popular_exit);

extern const int* get_popular_exit();


extern int h_env_popular_exit;


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#if defined(PROFILE)
#include "nvToolsExt.h"

#define PROFILE_WHITE   0x00eeeeee
#define PROFILE_GREEN   0x0000ff00
#define PROFILE_BLUE    0x000000ff
#define PROFILE_YELLOW  0x00ffff00
#define PROFILE_MAGENTA 0x00ff00ff
#define PROFILE_CYAN    0x0000ffff
#define PROFILE_RED     0x00ff0000
#define PROFILE_GREY    0x00999999
#define PROFILE_LILAC   0xC8A2C8

const uint32_t profile_colors[] = {
  PROFILE_WHITE,
  PROFILE_GREEN,
  PROFILE_BLUE,
  PROFILE_YELLOW,
  PROFILE_MAGENTA,
  PROFILE_CYAN,
  PROFILE_RED,
  PROFILE_GREY,
  PROFILE_LILAC
};
const int num_profile_colors = sizeof(profile_colors) / sizeof(uint32_t);

// Externed value containing colour information.
extern unsigned int g_profile_colour_id;

#define PROFILE_PUSH_RANGE(name) { \
    unsigned int color_id = g_profile_colour_id % num_profile_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = profile_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
    g_profile_colour_id++; \
}
#define PROFILE_POP_RANGE() nvtxRangePop();

// Class for simple fire-and-forget profile ranges (ie. functions with multiple return conditions.)
class ProfileScopedRange {
public:
    ProfileScopedRange(const char * name){
      PROFILE_PUSH_RANGE(name);
    }
    ~ProfileScopedRange(){
      PROFILE_POP_RANGE();
    }
};
#define PROFILE_SCOPED_RANGE(name) ProfileScopedRange uniq_name_using_macros(name);
#else
#define PROFILE_PUSH_RANGE(name)
#define PROFILE_POP_RANGE()
#define PROFILE_SCOPED_RANGE(name)
#endif


#endif //__HEADER

