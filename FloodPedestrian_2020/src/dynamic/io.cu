
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


#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <limits.h>
#include <algorithm>
#include <string>
#include <vector>



#ifdef _WIN32
#define strtok_r strtok_s
#endif

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

int fpgu_strtol(const char* str){
    return (int)strtol(str, NULL, 0);
}

unsigned int fpgu_strtoul(const char* str){
    return (unsigned int)strtoul(str, NULL, 0);
}

long long int fpgu_strtoll(const char* str){
    return strtoll(str, NULL, 0);
}

unsigned long long int fpgu_strtoull(const char* str){
    return strtoull(str, NULL, 0);
}

double fpgu_strtod(const char* str){
    return strtod(str, NULL);
}

float fgpu_atof(const char* str){
    return (float)atof(str);
}


//templated class function to read array inputs from supported types
template <class T>
void readArrayInput( T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: variable array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (T)parseFunc(token);
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: variable array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

//templated class function to read array inputs from supported types
template <class T, class BASE_T, unsigned int D>
void readArrayInputVectorType( BASE_T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = "|";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        //read vector type as an array
        T vec;
        readArrayInput<BASE_T>(parseFunc, token, (BASE_T*) &vec, D);
        array[i++] = vec;
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_FloodCell_list* h_FloodCells_Default, xmachine_memory_FloodCell_list* d_FloodCells_Default, int h_xmachine_memory_FloodCell_Default_count,xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count,xmachine_memory_navmap_list* h_navmaps_static, xmachine_memory_navmap_list* d_navmaps_static, int h_xmachine_memory_navmap_static_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_FloodCells_Default, d_FloodCells_Default, sizeof(xmachine_memory_FloodCell_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying FloodCell Agent Default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_agents_default, d_agents_default, sizeof(xmachine_memory_agent_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying agent Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_navmaps_static, d_navmaps_static, sizeof(xmachine_memory_navmap_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying navmap Agent static State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
    if(file == nullptr){
        printf("Error: Could not open file `%s` for output. Aborting.\n", data);
        exit(EXIT_FAILURE);
    }
    fputs("<states>\n<itno>", file);
    sprintf(data, "%i", iteration_number);
    fputs(data, file);
    fputs("</itno>\n", file);
    fputs("<environment>\n" , file);
    
    fputs("\t<outputting_time>", file);
    sprintf(data, "%f", (*get_outputting_time()));
    fputs(data, file);
    fputs("</outputting_time>\n", file);
    fputs("\t<outputting_time_interval>", file);
    sprintf(data, "%f", (*get_outputting_time_interval()));
    fputs(data, file);
    fputs("</outputting_time_interval>\n", file);
    fputs("\t<xmin>", file);
    sprintf(data, "%f", (*get_xmin()));
    fputs(data, file);
    fputs("</xmin>\n", file);
    fputs("\t<xmax>", file);
    sprintf(data, "%f", (*get_xmax()));
    fputs(data, file);
    fputs("</xmax>\n", file);
    fputs("\t<ymin>", file);
    sprintf(data, "%f", (*get_ymin()));
    fputs(data, file);
    fputs("</ymin>\n", file);
    fputs("\t<ymax>", file);
    sprintf(data, "%f", (*get_ymax()));
    fputs(data, file);
    fputs("</ymax>\n", file);
    fputs("\t<dt_ped>", file);
    sprintf(data, "%f", (*get_dt_ped()));
    fputs(data, file);
    fputs("</dt_ped>\n", file);
    fputs("\t<dt_flood>", file);
    sprintf(data, "%f", (*get_dt_flood()));
    fputs(data, file);
    fputs("</dt_flood>\n", file);
    fputs("\t<dt>", file);
    sprintf(data, "%f", (*get_dt()));
    fputs(data, file);
    fputs("</dt>\n", file);
    fputs("\t<auto_dt_on>", file);
    sprintf(data, "%d", (*get_auto_dt_on()));
    fputs(data, file);
    fputs("</auto_dt_on>\n", file);
    fputs("\t<body_as_obstacle_on>", file);
    sprintf(data, "%d", (*get_body_as_obstacle_on()));
    fputs(data, file);
    fputs("</body_as_obstacle_on>\n", file);
    fputs("\t<ped_roughness_effect_on>", file);
    sprintf(data, "%d", (*get_ped_roughness_effect_on()));
    fputs(data, file);
    fputs("</ped_roughness_effect_on>\n", file);
    fputs("\t<body_height>", file);
    sprintf(data, "%f", (*get_body_height()));
    fputs(data, file);
    fputs("</body_height>\n", file);
    fputs("\t<init_speed>", file);
    sprintf(data, "%f", (*get_init_speed()));
    fputs(data, file);
    fputs("</init_speed>\n", file);
    fputs("\t<brisk_speed>", file);
    sprintf(data, "%f", (*get_brisk_speed()));
    fputs(data, file);
    fputs("</brisk_speed>\n", file);
    fputs("\t<sim_time>", file);
    sprintf(data, "%f", (*get_sim_time()));
    fputs(data, file);
    fputs("</sim_time>\n", file);
    fputs("\t<DXL>", file);
    sprintf(data, "%f", (*get_DXL()));
    fputs(data, file);
    fputs("</DXL>\n", file);
    fputs("\t<DYL>", file);
    sprintf(data, "%f", (*get_DYL()));
    fputs(data, file);
    fputs("</DYL>\n", file);
    fputs("\t<inflow_start_time>", file);
    sprintf(data, "%f", (*get_inflow_start_time()));
    fputs(data, file);
    fputs("</inflow_start_time>\n", file);
    fputs("\t<inflow_peak_time>", file);
    sprintf(data, "%f", (*get_inflow_peak_time()));
    fputs(data, file);
    fputs("</inflow_peak_time>\n", file);
    fputs("\t<inflow_end_time>", file);
    sprintf(data, "%f", (*get_inflow_end_time()));
    fputs(data, file);
    fputs("</inflow_end_time>\n", file);
    fputs("\t<inflow_initial_discharge>", file);
    sprintf(data, "%f", (*get_inflow_initial_discharge()));
    fputs(data, file);
    fputs("</inflow_initial_discharge>\n", file);
    fputs("\t<inflow_peak_discharge>", file);
    sprintf(data, "%f", (*get_inflow_peak_discharge()));
    fputs(data, file);
    fputs("</inflow_peak_discharge>\n", file);
    fputs("\t<inflow_end_discharge>", file);
    sprintf(data, "%f", (*get_inflow_end_discharge()));
    fputs(data, file);
    fputs("</inflow_end_discharge>\n", file);
    fputs("\t<INFLOW_BOUNDARY>", file);
    sprintf(data, "%d", (*get_INFLOW_BOUNDARY()));
    fputs(data, file);
    fputs("</INFLOW_BOUNDARY>\n", file);
    fputs("\t<BOUNDARY_EAST_STATUS>", file);
    sprintf(data, "%d", (*get_BOUNDARY_EAST_STATUS()));
    fputs(data, file);
    fputs("</BOUNDARY_EAST_STATUS>\n", file);
    fputs("\t<BOUNDARY_WEST_STATUS>", file);
    sprintf(data, "%d", (*get_BOUNDARY_WEST_STATUS()));
    fputs(data, file);
    fputs("</BOUNDARY_WEST_STATUS>\n", file);
    fputs("\t<BOUNDARY_NORTH_STATUS>", file);
    sprintf(data, "%d", (*get_BOUNDARY_NORTH_STATUS()));
    fputs(data, file);
    fputs("</BOUNDARY_NORTH_STATUS>\n", file);
    fputs("\t<BOUNDARY_SOUTH_STATUS>", file);
    sprintf(data, "%d", (*get_BOUNDARY_SOUTH_STATUS()));
    fputs(data, file);
    fputs("</BOUNDARY_SOUTH_STATUS>\n", file);
    fputs("\t<x1_boundary>", file);
    sprintf(data, "%f", (*get_x1_boundary()));
    fputs(data, file);
    fputs("</x1_boundary>\n", file);
    fputs("\t<x2_boundary>", file);
    sprintf(data, "%f", (*get_x2_boundary()));
    fputs(data, file);
    fputs("</x2_boundary>\n", file);
    fputs("\t<y1_boundary>", file);
    sprintf(data, "%f", (*get_y1_boundary()));
    fputs(data, file);
    fputs("</y1_boundary>\n", file);
    fputs("\t<y2_boundary>", file);
    sprintf(data, "%f", (*get_y2_boundary()));
    fputs(data, file);
    fputs("</y2_boundary>\n", file);
    fputs("\t<init_depth_boundary>", file);
    sprintf(data, "%f", (*get_init_depth_boundary()));
    fputs(data, file);
    fputs("</init_depth_boundary>\n", file);
    fputs("\t<evacuation_on>", file);
    sprintf(data, "%d", (*get_evacuation_on()));
    fputs(data, file);
    fputs("</evacuation_on>\n", file);
    fputs("\t<walking_speed_reduction_in_water_on>", file);
    sprintf(data, "%d", (*get_walking_speed_reduction_in_water_on()));
    fputs(data, file);
    fputs("</walking_speed_reduction_in_water_on>\n", file);
    fputs("\t<freeze_while_instable_on>", file);
    sprintf(data, "%d", (*get_freeze_while_instable_on()));
    fputs(data, file);
    fputs("</freeze_while_instable_on>\n", file);
    fputs("\t<evacuation_end_time>", file);
    sprintf(data, "%f", (*get_evacuation_end_time()));
    fputs(data, file);
    fputs("</evacuation_end_time>\n", file);
    fputs("\t<evacuation_start_time>", file);
    sprintf(data, "%f", (*get_evacuation_start_time()));
    fputs(data, file);
    fputs("</evacuation_start_time>\n", file);
    fputs("\t<emergency_exit_number>", file);
    sprintf(data, "%d", (*get_emergency_exit_number()));
    fputs(data, file);
    fputs("</emergency_exit_number>\n", file);
    fputs("\t<emer_alarm>", file);
    sprintf(data, "%d", (*get_emer_alarm()));
    fputs(data, file);
    fputs("</emer_alarm>\n", file);
    fputs("\t<HR>", file);
    sprintf(data, "%f", (*get_HR()));
    fputs(data, file);
    fputs("</HR>\n", file);
    fputs("\t<max_at_highest_risk>", file);
    sprintf(data, "%d", (*get_max_at_highest_risk()));
    fputs(data, file);
    fputs("</max_at_highest_risk>\n", file);
    fputs("\t<max_at_low_risk>", file);
    sprintf(data, "%d", (*get_max_at_low_risk()));
    fputs(data, file);
    fputs("</max_at_low_risk>\n", file);
    fputs("\t<max_at_medium_risk>", file);
    sprintf(data, "%d", (*get_max_at_medium_risk()));
    fputs(data, file);
    fputs("</max_at_medium_risk>\n", file);
    fputs("\t<max_at_high_risk>", file);
    sprintf(data, "%d", (*get_max_at_high_risk()));
    fputs(data, file);
    fputs("</max_at_high_risk>\n", file);
    fputs("\t<max_velocity>", file);
    sprintf(data, "%f", (*get_max_velocity()));
    fputs(data, file);
    fputs("</max_velocity>\n", file);
    fputs("\t<max_depth>", file);
    sprintf(data, "%f", (*get_max_depth()));
    fputs(data, file);
    fputs("</max_depth>\n", file);
    fputs("\t<count_population>", file);
    sprintf(data, "%d", (*get_count_population()));
    fputs(data, file);
    fputs("</count_population>\n", file);
    fputs("\t<count_heros>", file);
    sprintf(data, "%d", (*get_count_heros()));
    fputs(data, file);
    fputs("</count_heros>\n", file);
    fputs("\t<initial_population>", file);
    sprintf(data, "%d", (*get_initial_population()));
    fputs(data, file);
    fputs("</initial_population>\n", file);
    fputs("\t<evacuated_population>", file);
    sprintf(data, "%d", (*get_evacuated_population()));
    fputs(data, file);
    fputs("</evacuated_population>\n", file);
    fputs("\t<hero_percentage>", file);
    sprintf(data, "%f", (*get_hero_percentage()));
    fputs(data, file);
    fputs("</hero_percentage>\n", file);
    fputs("\t<hero_population>", file);
    sprintf(data, "%d", (*get_hero_population()));
    fputs(data, file);
    fputs("</hero_population>\n", file);
    fputs("\t<sandbagging_on>", file);
    sprintf(data, "%d", (*get_sandbagging_on()));
    fputs(data, file);
    fputs("</sandbagging_on>\n", file);
    fputs("\t<sandbagging_start_time>", file);
    sprintf(data, "%f", (*get_sandbagging_start_time()));
    fputs(data, file);
    fputs("</sandbagging_start_time>\n", file);
    fputs("\t<sandbagging_end_time>", file);
    sprintf(data, "%f", (*get_sandbagging_end_time()));
    fputs(data, file);
    fputs("</sandbagging_end_time>\n", file);
    fputs("\t<sandbag_length>", file);
    sprintf(data, "%f", (*get_sandbag_length()));
    fputs(data, file);
    fputs("</sandbag_length>\n", file);
    fputs("\t<sandbag_height>", file);
    sprintf(data, "%f", (*get_sandbag_height()));
    fputs(data, file);
    fputs("</sandbag_height>\n", file);
    fputs("\t<sandbag_width>", file);
    sprintf(data, "%f", (*get_sandbag_width()));
    fputs(data, file);
    fputs("</sandbag_width>\n", file);
    fputs("\t<extended_length>", file);
    sprintf(data, "%f", (*get_extended_length()));
    fputs(data, file);
    fputs("</extended_length>\n", file);
    fputs("\t<sandbag_layers>", file);
    sprintf(data, "%d", (*get_sandbag_layers()));
    fputs(data, file);
    fputs("</sandbag_layers>\n", file);
    fputs("\t<update_stopper>", file);
    sprintf(data, "%d", (*get_update_stopper()));
    fputs(data, file);
    fputs("</update_stopper>\n", file);
    fputs("\t<dike_length>", file);
    sprintf(data, "%f", (*get_dike_length()));
    fputs(data, file);
    fputs("</dike_length>\n", file);
    fputs("\t<dike_height>", file);
    sprintf(data, "%f", (*get_dike_height()));
    fputs(data, file);
    fputs("</dike_height>\n", file);
    fputs("\t<dike_width>", file);
    sprintf(data, "%f", (*get_dike_width()));
    fputs(data, file);
    fputs("</dike_width>\n", file);
    fputs("\t<fill_cap>", file);
    sprintf(data, "%d", (*get_fill_cap()));
    fputs(data, file);
    fputs("</fill_cap>\n", file);
    fputs("\t<pickup_point>", file);
    sprintf(data, "%d", (*get_pickup_point()));
    fputs(data, file);
    fputs("</pickup_point>\n", file);
    fputs("\t<drop_point>", file);
    sprintf(data, "%d", (*get_drop_point()));
    fputs(data, file);
    fputs("</drop_point>\n", file);
    fputs("\t<pickup_duration>", file);
    sprintf(data, "%f", (*get_pickup_duration()));
    fputs(data, file);
    fputs("</pickup_duration>\n", file);
    fputs("\t<drop_duration>", file);
    sprintf(data, "%f", (*get_drop_duration()));
    fputs(data, file);
    fputs("</drop_duration>\n", file);
    fputs("\t<EMMISION_RATE_EXIT1>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT1()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT1>\n", file);
    fputs("\t<EMMISION_RATE_EXIT2>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT2()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT2>\n", file);
    fputs("\t<EMMISION_RATE_EXIT3>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT3()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT3>\n", file);
    fputs("\t<EMMISION_RATE_EXIT4>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT4()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT4>\n", file);
    fputs("\t<EMMISION_RATE_EXIT5>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT5()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT5>\n", file);
    fputs("\t<EMMISION_RATE_EXIT6>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT6()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT6>\n", file);
    fputs("\t<EMMISION_RATE_EXIT7>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT7()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT7>\n", file);
    fputs("\t<EMMISION_RATE_EXIT8>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT8()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT8>\n", file);
    fputs("\t<EMMISION_RATE_EXIT9>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT9()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT9>\n", file);
    fputs("\t<EMMISION_RATE_EXIT10>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT10()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT10>\n", file);
    fputs("\t<EXIT1_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT1_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT1_PROBABILITY>\n", file);
    fputs("\t<EXIT2_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT2_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT2_PROBABILITY>\n", file);
    fputs("\t<EXIT3_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT3_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT3_PROBABILITY>\n", file);
    fputs("\t<EXIT4_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT4_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT4_PROBABILITY>\n", file);
    fputs("\t<EXIT5_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT5_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT5_PROBABILITY>\n", file);
    fputs("\t<EXIT6_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT6_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT6_PROBABILITY>\n", file);
    fputs("\t<EXIT7_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT7_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT7_PROBABILITY>\n", file);
    fputs("\t<EXIT8_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT8_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT8_PROBABILITY>\n", file);
    fputs("\t<EXIT9_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT9_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT9_PROBABILITY>\n", file);
    fputs("\t<EXIT10_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT10_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT10_PROBABILITY>\n", file);
    fputs("\t<EXIT1_STATE>", file);
    sprintf(data, "%d", (*get_EXIT1_STATE()));
    fputs(data, file);
    fputs("</EXIT1_STATE>\n", file);
    fputs("\t<EXIT2_STATE>", file);
    sprintf(data, "%d", (*get_EXIT2_STATE()));
    fputs(data, file);
    fputs("</EXIT2_STATE>\n", file);
    fputs("\t<EXIT3_STATE>", file);
    sprintf(data, "%d", (*get_EXIT3_STATE()));
    fputs(data, file);
    fputs("</EXIT3_STATE>\n", file);
    fputs("\t<EXIT4_STATE>", file);
    sprintf(data, "%d", (*get_EXIT4_STATE()));
    fputs(data, file);
    fputs("</EXIT4_STATE>\n", file);
    fputs("\t<EXIT5_STATE>", file);
    sprintf(data, "%d", (*get_EXIT5_STATE()));
    fputs(data, file);
    fputs("</EXIT5_STATE>\n", file);
    fputs("\t<EXIT6_STATE>", file);
    sprintf(data, "%d", (*get_EXIT6_STATE()));
    fputs(data, file);
    fputs("</EXIT6_STATE>\n", file);
    fputs("\t<EXIT7_STATE>", file);
    sprintf(data, "%d", (*get_EXIT7_STATE()));
    fputs(data, file);
    fputs("</EXIT7_STATE>\n", file);
    fputs("\t<EXIT8_STATE>", file);
    sprintf(data, "%d", (*get_EXIT8_STATE()));
    fputs(data, file);
    fputs("</EXIT8_STATE>\n", file);
    fputs("\t<EXIT9_STATE>", file);
    sprintf(data, "%d", (*get_EXIT9_STATE()));
    fputs(data, file);
    fputs("</EXIT9_STATE>\n", file);
    fputs("\t<EXIT10_STATE>", file);
    sprintf(data, "%d", (*get_EXIT10_STATE()));
    fputs(data, file);
    fputs("</EXIT10_STATE>\n", file);
    fputs("\t<EXIT1_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT1_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT1_CELL_COUNT>\n", file);
    fputs("\t<EXIT2_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT2_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT2_CELL_COUNT>\n", file);
    fputs("\t<EXIT3_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT3_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT3_CELL_COUNT>\n", file);
    fputs("\t<EXIT4_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT4_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT4_CELL_COUNT>\n", file);
    fputs("\t<EXIT5_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT5_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT5_CELL_COUNT>\n", file);
    fputs("\t<EXIT6_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT6_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT6_CELL_COUNT>\n", file);
    fputs("\t<EXIT7_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT7_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT7_CELL_COUNT>\n", file);
    fputs("\t<EXIT8_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT8_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT8_CELL_COUNT>\n", file);
    fputs("\t<EXIT9_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT9_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT9_CELL_COUNT>\n", file);
    fputs("\t<EXIT10_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT10_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT10_CELL_COUNT>\n", file);
    fputs("\t<TIME_SCALER>", file);
    sprintf(data, "%f", (*get_TIME_SCALER()));
    fputs(data, file);
    fputs("</TIME_SCALER>\n", file);
    fputs("\t<STEER_WEIGHT>", file);
    sprintf(data, "%f", (*get_STEER_WEIGHT()));
    fputs(data, file);
    fputs("</STEER_WEIGHT>\n", file);
    fputs("\t<AVOID_WEIGHT>", file);
    sprintf(data, "%f", (*get_AVOID_WEIGHT()));
    fputs(data, file);
    fputs("</AVOID_WEIGHT>\n", file);
    fputs("\t<COLLISION_WEIGHT>", file);
    sprintf(data, "%f", (*get_COLLISION_WEIGHT()));
    fputs(data, file);
    fputs("</COLLISION_WEIGHT>\n", file);
    fputs("\t<GOAL_WEIGHT>", file);
    sprintf(data, "%f", (*get_GOAL_WEIGHT()));
    fputs(data, file);
    fputs("</GOAL_WEIGHT>\n", file);
    fputs("\t<PedHeight_60_110_probability>", file);
    sprintf(data, "%d", (*get_PedHeight_60_110_probability()));
    fputs(data, file);
    fputs("</PedHeight_60_110_probability>\n", file);
    fputs("\t<PedHeight_110_140_probability>", file);
    sprintf(data, "%d", (*get_PedHeight_110_140_probability()));
    fputs(data, file);
    fputs("</PedHeight_110_140_probability>\n", file);
    fputs("\t<PedHeight_140_163_probability>", file);
    sprintf(data, "%d", (*get_PedHeight_140_163_probability()));
    fputs(data, file);
    fputs("</PedHeight_140_163_probability>\n", file);
    fputs("\t<PedHeight_163_170_probability>", file);
    sprintf(data, "%d", (*get_PedHeight_163_170_probability()));
    fputs(data, file);
    fputs("</PedHeight_163_170_probability>\n", file);
    fputs("\t<PedHeight_170_186_probability>", file);
    sprintf(data, "%d", (*get_PedHeight_170_186_probability()));
    fputs(data, file);
    fputs("</PedHeight_170_186_probability>\n", file);
    fputs("\t<PedHeight_186_194_probability>", file);
    sprintf(data, "%d", (*get_PedHeight_186_194_probability()));
    fputs(data, file);
    fputs("</PedHeight_186_194_probability>\n", file);
    fputs("\t<PedHeight_194_210_probability>", file);
    sprintf(data, "%d", (*get_PedHeight_194_210_probability()));
    fputs(data, file);
    fputs("</PedHeight_194_210_probability>\n", file);
    fputs("\t<PedAge_10_17_probability>", file);
    sprintf(data, "%d", (*get_PedAge_10_17_probability()));
    fputs(data, file);
    fputs("</PedAge_10_17_probability>\n", file);
    fputs("\t<PedAge_18_29_probability>", file);
    sprintf(data, "%d", (*get_PedAge_18_29_probability()));
    fputs(data, file);
    fputs("</PedAge_18_29_probability>\n", file);
    fputs("\t<PedAge_30_39_probability>", file);
    sprintf(data, "%d", (*get_PedAge_30_39_probability()));
    fputs(data, file);
    fputs("</PedAge_30_39_probability>\n", file);
    fputs("\t<PedAge_40_49_probability>", file);
    sprintf(data, "%d", (*get_PedAge_40_49_probability()));
    fputs(data, file);
    fputs("</PedAge_40_49_probability>\n", file);
    fputs("\t<PedAge_50_59_probability>", file);
    sprintf(data, "%d", (*get_PedAge_50_59_probability()));
    fputs(data, file);
    fputs("</PedAge_50_59_probability>\n", file);
    fputs("\t<PedAge_60_69_probability>", file);
    sprintf(data, "%d", (*get_PedAge_60_69_probability()));
    fputs(data, file);
    fputs("</PedAge_60_69_probability>\n", file);
    fputs("\t<PedAge_70_79_probability>", file);
    sprintf(data, "%d", (*get_PedAge_70_79_probability()));
    fputs(data, file);
    fputs("</PedAge_70_79_probability>\n", file);
    fputs("\t<excluded_age_probability>", file);
    sprintf(data, "%d", (*get_excluded_age_probability()));
    fputs(data, file);
    fputs("</excluded_age_probability>\n", file);
    fputs("\t<gender_female_probability>", file);
    sprintf(data, "%d", (*get_gender_female_probability()));
    fputs(data, file);
    fputs("</gender_female_probability>\n", file);
    fputs("\t<gender_male_probability>", file);
    sprintf(data, "%d", (*get_gender_male_probability()));
    fputs(data, file);
    fputs("</gender_male_probability>\n", file);
    fputs("\t<SCALE_FACTOR>", file);
    sprintf(data, "%f", (*get_SCALE_FACTOR()));
    fputs(data, file);
    fputs("</SCALE_FACTOR>\n", file);
    fputs("\t<I_SCALER>", file);
    sprintf(data, "%f", (*get_I_SCALER()));
    fputs(data, file);
    fputs("</I_SCALER>\n", file);
    fputs("\t<MIN_DISTANCE>", file);
    sprintf(data, "%f", (*get_MIN_DISTANCE()));
    fputs(data, file);
    fputs("</MIN_DISTANCE>\n", file);
    fputs("\t<excitement_on>", file);
    sprintf(data, "%d", (*get_excitement_on()));
    fputs(data, file);
    fputs("</excitement_on>\n", file);
    fputs("\t<walk_run_switch>", file);
    sprintf(data, "%d", (*get_walk_run_switch()));
    fputs(data, file);
    fputs("</walk_run_switch>\n", file);
    fputs("\t<preoccupying_on>", file);
    sprintf(data, "%d", (*get_preoccupying_on()));
    fputs(data, file);
    fputs("</preoccupying_on>\n", file);
    fputs("\t<poly_hydrograph_on>", file);
    sprintf(data, "%d", (*get_poly_hydrograph_on()));
    fputs(data, file);
    fputs("</poly_hydrograph_on>\n", file);
    fputs("\t<stop_emission_on>", file);
    sprintf(data, "%d", (*get_stop_emission_on()));
    fputs(data, file);
    fputs("</stop_emission_on>\n", file);
    fputs("\t<goto_emergency_exit_on>", file);
    sprintf(data, "%d", (*get_goto_emergency_exit_on()));
    fputs(data, file);
    fputs("</goto_emergency_exit_on>\n", file);
    fputs("\t<escape_route_finder_on>", file);
    sprintf(data, "%d", (*get_escape_route_finder_on()));
    fputs(data, file);
    fputs("</escape_route_finder_on>\n", file);
    fputs("\t<dir_times>", file);
    sprintf(data, "%d", (*get_dir_times()));
    fputs(data, file);
    fputs("</dir_times>\n", file);
    fputs("\t<no_return_on>", file);
    sprintf(data, "%d", (*get_no_return_on()));
    fputs(data, file);
    fputs("</no_return_on>\n", file);
    fputs("\t<wdepth_perc_thresh>", file);
    sprintf(data, "%f", (*get_wdepth_perc_thresh()));
    fputs(data, file);
    fputs("</wdepth_perc_thresh>\n", file);
    fputs("\t<follow_popular_exit_on>", file);
    sprintf(data, "%d", (*get_follow_popular_exit_on()));
    fputs(data, file);
    fputs("</follow_popular_exit_on>\n", file);
    fputs("\t<popular_exit>", file);
    sprintf(data, "%d", (*get_popular_exit()));
    fputs(data, file);
    fputs("</popular_exit>\n", file);
	fputs("</environment>\n" , file);

	//Write each FloodCell agent to xml
	for (int i=0; i<h_xmachine_memory_FloodCell_Default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>FloodCell</name>\n", file);
        
		fputs("<inDomain>", file);
        sprintf(data, "%d", h_FloodCells_Default->inDomain[i]);
		fputs(data, file);
		fputs("</inDomain>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%d", h_FloodCells_Default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%d", h_FloodCells_Default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z0>", file);
        sprintf(data, "%f", h_FloodCells_Default->z0[i]);
		fputs(data, file);
		fputs("</z0>\n", file);
        
		fputs("<h>", file);
        sprintf(data, "%f", h_FloodCells_Default->h[i]);
		fputs(data, file);
		fputs("</h>\n", file);
        
		fputs("<qx>", file);
        sprintf(data, "%f", h_FloodCells_Default->qx[i]);
		fputs(data, file);
		fputs("</qx>\n", file);
        
		fputs("<qy>", file);
        sprintf(data, "%f", h_FloodCells_Default->qy[i]);
		fputs(data, file);
		fputs("</qy>\n", file);
        
		fputs("<timeStep>", file);
        sprintf(data, "%f", h_FloodCells_Default->timeStep[i]);
		fputs(data, file);
		fputs("</timeStep>\n", file);
        
		fputs("<minh_loc>", file);
        sprintf(data, "%f", h_FloodCells_Default->minh_loc[i]);
		fputs(data, file);
		fputs("</minh_loc>\n", file);
        
		fputs("<hFace_E>", file);
        sprintf(data, "%f", h_FloodCells_Default->hFace_E[i]);
		fputs(data, file);
		fputs("</hFace_E>\n", file);
        
		fputs("<etFace_E>", file);
        sprintf(data, "%f", h_FloodCells_Default->etFace_E[i]);
		fputs(data, file);
		fputs("</etFace_E>\n", file);
        
		fputs("<qxFace_E>", file);
        sprintf(data, "%f", h_FloodCells_Default->qxFace_E[i]);
		fputs(data, file);
		fputs("</qxFace_E>\n", file);
        
		fputs("<qyFace_E>", file);
        sprintf(data, "%f", h_FloodCells_Default->qyFace_E[i]);
		fputs(data, file);
		fputs("</qyFace_E>\n", file);
        
		fputs("<hFace_W>", file);
        sprintf(data, "%f", h_FloodCells_Default->hFace_W[i]);
		fputs(data, file);
		fputs("</hFace_W>\n", file);
        
		fputs("<etFace_W>", file);
        sprintf(data, "%f", h_FloodCells_Default->etFace_W[i]);
		fputs(data, file);
		fputs("</etFace_W>\n", file);
        
		fputs("<qxFace_W>", file);
        sprintf(data, "%f", h_FloodCells_Default->qxFace_W[i]);
		fputs(data, file);
		fputs("</qxFace_W>\n", file);
        
		fputs("<qyFace_W>", file);
        sprintf(data, "%f", h_FloodCells_Default->qyFace_W[i]);
		fputs(data, file);
		fputs("</qyFace_W>\n", file);
        
		fputs("<hFace_N>", file);
        sprintf(data, "%f", h_FloodCells_Default->hFace_N[i]);
		fputs(data, file);
		fputs("</hFace_N>\n", file);
        
		fputs("<etFace_N>", file);
        sprintf(data, "%f", h_FloodCells_Default->etFace_N[i]);
		fputs(data, file);
		fputs("</etFace_N>\n", file);
        
		fputs("<qxFace_N>", file);
        sprintf(data, "%f", h_FloodCells_Default->qxFace_N[i]);
		fputs(data, file);
		fputs("</qxFace_N>\n", file);
        
		fputs("<qyFace_N>", file);
        sprintf(data, "%f", h_FloodCells_Default->qyFace_N[i]);
		fputs(data, file);
		fputs("</qyFace_N>\n", file);
        
		fputs("<hFace_S>", file);
        sprintf(data, "%f", h_FloodCells_Default->hFace_S[i]);
		fputs(data, file);
		fputs("</hFace_S>\n", file);
        
		fputs("<etFace_S>", file);
        sprintf(data, "%f", h_FloodCells_Default->etFace_S[i]);
		fputs(data, file);
		fputs("</etFace_S>\n", file);
        
		fputs("<qxFace_S>", file);
        sprintf(data, "%f", h_FloodCells_Default->qxFace_S[i]);
		fputs(data, file);
		fputs("</qxFace_S>\n", file);
        
		fputs("<qyFace_S>", file);
        sprintf(data, "%f", h_FloodCells_Default->qyFace_S[i]);
		fputs(data, file);
		fputs("</qyFace_S>\n", file);
        
		fputs("<nm_rough>", file);
        sprintf(data, "%f", h_FloodCells_Default->nm_rough[i]);
		fputs(data, file);
		fputs("</nm_rough>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each agent agent to xml
	for (int i=0; i<h_xmachine_memory_agent_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>agent</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_agents_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_agents_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<velx>", file);
        sprintf(data, "%f", h_agents_default->velx[i]);
		fputs(data, file);
		fputs("</velx>\n", file);
        
		fputs("<vely>", file);
        sprintf(data, "%f", h_agents_default->vely[i]);
		fputs(data, file);
		fputs("</vely>\n", file);
        
		fputs("<steer_x>", file);
        sprintf(data, "%f", h_agents_default->steer_x[i]);
		fputs(data, file);
		fputs("</steer_x>\n", file);
        
		fputs("<steer_y>", file);
        sprintf(data, "%f", h_agents_default->steer_y[i]);
		fputs(data, file);
		fputs("</steer_y>\n", file);
        
		fputs("<height>", file);
        sprintf(data, "%f", h_agents_default->height[i]);
		fputs(data, file);
		fputs("</height>\n", file);
        
		fputs("<exit_no>", file);
        sprintf(data, "%d", h_agents_default->exit_no[i]);
		fputs(data, file);
		fputs("</exit_no>\n", file);
        
		fputs("<speed>", file);
        sprintf(data, "%f", h_agents_default->speed[i]);
		fputs(data, file);
		fputs("</speed>\n", file);
        
		fputs("<lod>", file);
        sprintf(data, "%d", h_agents_default->lod[i]);
		fputs(data, file);
		fputs("</lod>\n", file);
        
		fputs("<animate>", file);
        sprintf(data, "%f", h_agents_default->animate[i]);
		fputs(data, file);
		fputs("</animate>\n", file);
        
		fputs("<animate_dir>", file);
        sprintf(data, "%d", h_agents_default->animate_dir[i]);
		fputs(data, file);
		fputs("</animate_dir>\n", file);
        
		fputs("<HR_state>", file);
        sprintf(data, "%d", h_agents_default->HR_state[i]);
		fputs(data, file);
		fputs("</HR_state>\n", file);
        
		fputs("<hero_status>", file);
        sprintf(data, "%d", h_agents_default->hero_status[i]);
		fputs(data, file);
		fputs("</hero_status>\n", file);
        
		fputs("<pickup_time>", file);
        sprintf(data, "%f", h_agents_default->pickup_time[i]);
		fputs(data, file);
		fputs("</pickup_time>\n", file);
        
		fputs("<drop_time>", file);
        sprintf(data, "%f", h_agents_default->drop_time[i]);
		fputs(data, file);
		fputs("</drop_time>\n", file);
        
		fputs("<carry_sandbag>", file);
        sprintf(data, "%d", h_agents_default->carry_sandbag[i]);
		fputs(data, file);
		fputs("</carry_sandbag>\n", file);
        
		fputs("<HR>", file);
        sprintf(data, "%f", h_agents_default->HR[i]);
		fputs(data, file);
		fputs("</HR>\n", file);
        
		fputs("<dt_ped>", file);
        sprintf(data, "%f", h_agents_default->dt_ped[i]);
		fputs(data, file);
		fputs("</dt_ped>\n", file);
        
		fputs("<d_water>", file);
        sprintf(data, "%f", h_agents_default->d_water[i]);
		fputs(data, file);
		fputs("</d_water>\n", file);
        
		fputs("<v_water>", file);
        sprintf(data, "%f", h_agents_default->v_water[i]);
		fputs(data, file);
		fputs("</v_water>\n", file);
        
		fputs("<body_height>", file);
        sprintf(data, "%f", h_agents_default->body_height[i]);
		fputs(data, file);
		fputs("</body_height>\n", file);
        
		fputs("<body_mass>", file);
        sprintf(data, "%f", h_agents_default->body_mass[i]);
		fputs(data, file);
		fputs("</body_mass>\n", file);
        
		fputs("<gender>", file);
        sprintf(data, "%d", h_agents_default->gender[i]);
		fputs(data, file);
		fputs("</gender>\n", file);
        
		fputs("<stability_state>", file);
        sprintf(data, "%d", h_agents_default->stability_state[i]);
		fputs(data, file);
		fputs("</stability_state>\n", file);
        
		fputs("<motion_speed>", file);
        sprintf(data, "%f", h_agents_default->motion_speed[i]);
		fputs(data, file);
		fputs("</motion_speed>\n", file);
        
		fputs("<age>", file);
        sprintf(data, "%d", h_agents_default->age[i]);
		fputs(data, file);
		fputs("</age>\n", file);
        
		fputs("<excitement_speed>", file);
        sprintf(data, "%f", h_agents_default->excitement_speed[i]);
		fputs(data, file);
		fputs("</excitement_speed>\n", file);
        
		fputs("<dir_times>", file);
        sprintf(data, "%d", h_agents_default->dir_times[i]);
		fputs(data, file);
		fputs("</dir_times>\n", file);
        
		fputs("<rejected_exit1>", file);
        sprintf(data, "%d", h_agents_default->rejected_exit1[i]);
		fputs(data, file);
		fputs("</rejected_exit1>\n", file);
        
		fputs("<rejected_exit2>", file);
        sprintf(data, "%d", h_agents_default->rejected_exit2[i]);
		fputs(data, file);
		fputs("</rejected_exit2>\n", file);
        
		fputs("<rejected_exit3>", file);
        sprintf(data, "%d", h_agents_default->rejected_exit3[i]);
		fputs(data, file);
		fputs("</rejected_exit3>\n", file);
        
		fputs("<rejected_exit4>", file);
        sprintf(data, "%d", h_agents_default->rejected_exit4[i]);
		fputs(data, file);
		fputs("</rejected_exit4>\n", file);
        
		fputs("<rejected_exit5>", file);
        sprintf(data, "%d", h_agents_default->rejected_exit5[i]);
		fputs(data, file);
		fputs("</rejected_exit5>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each navmap agent to xml
	for (int i=0; i<h_xmachine_memory_navmap_static_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>navmap</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%d", h_navmaps_static->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%d", h_navmaps_static->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z0>", file);
        sprintf(data, "%f", h_navmaps_static->z0[i]);
		fputs(data, file);
		fputs("</z0>\n", file);
        
		fputs("<h>", file);
        sprintf(data, "%f", h_navmaps_static->h[i]);
		fputs(data, file);
		fputs("</h>\n", file);
        
		fputs("<qx>", file);
        sprintf(data, "%f", h_navmaps_static->qx[i]);
		fputs(data, file);
		fputs("</qx>\n", file);
        
		fputs("<qy>", file);
        sprintf(data, "%f", h_navmaps_static->qy[i]);
		fputs(data, file);
		fputs("</qy>\n", file);
        
		fputs("<exit_no>", file);
        sprintf(data, "%d", h_navmaps_static->exit_no[i]);
		fputs(data, file);
		fputs("</exit_no>\n", file);
        
		fputs("<height>", file);
        sprintf(data, "%f", h_navmaps_static->height[i]);
		fputs(data, file);
		fputs("</height>\n", file);
        
		fputs("<collision_x>", file);
        sprintf(data, "%f", h_navmaps_static->collision_x[i]);
		fputs(data, file);
		fputs("</collision_x>\n", file);
        
		fputs("<collision_y>", file);
        sprintf(data, "%f", h_navmaps_static->collision_y[i]);
		fputs(data, file);
		fputs("</collision_y>\n", file);
        
		fputs("<exit0_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit0_x[i]);
		fputs(data, file);
		fputs("</exit0_x>\n", file);
        
		fputs("<exit0_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit0_y[i]);
		fputs(data, file);
		fputs("</exit0_y>\n", file);
        
		fputs("<exit1_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit1_x[i]);
		fputs(data, file);
		fputs("</exit1_x>\n", file);
        
		fputs("<exit1_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit1_y[i]);
		fputs(data, file);
		fputs("</exit1_y>\n", file);
        
		fputs("<exit2_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit2_x[i]);
		fputs(data, file);
		fputs("</exit2_x>\n", file);
        
		fputs("<exit2_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit2_y[i]);
		fputs(data, file);
		fputs("</exit2_y>\n", file);
        
		fputs("<exit3_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit3_x[i]);
		fputs(data, file);
		fputs("</exit3_x>\n", file);
        
		fputs("<exit3_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit3_y[i]);
		fputs(data, file);
		fputs("</exit3_y>\n", file);
        
		fputs("<exit4_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit4_x[i]);
		fputs(data, file);
		fputs("</exit4_x>\n", file);
        
		fputs("<exit4_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit4_y[i]);
		fputs(data, file);
		fputs("</exit4_y>\n", file);
        
		fputs("<exit5_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit5_x[i]);
		fputs(data, file);
		fputs("</exit5_x>\n", file);
        
		fputs("<exit5_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit5_y[i]);
		fputs(data, file);
		fputs("</exit5_y>\n", file);
        
		fputs("<exit6_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit6_x[i]);
		fputs(data, file);
		fputs("</exit6_x>\n", file);
        
		fputs("<exit6_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit6_y[i]);
		fputs(data, file);
		fputs("</exit6_y>\n", file);
        
		fputs("<exit7_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit7_x[i]);
		fputs(data, file);
		fputs("</exit7_x>\n", file);
        
		fputs("<exit7_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit7_y[i]);
		fputs(data, file);
		fputs("</exit7_y>\n", file);
        
		fputs("<exit8_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit8_x[i]);
		fputs(data, file);
		fputs("</exit8_x>\n", file);
        
		fputs("<exit8_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit8_y[i]);
		fputs(data, file);
		fputs("</exit8_y>\n", file);
        
		fputs("<exit9_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit9_x[i]);
		fputs(data, file);
		fputs("</exit9_x>\n", file);
        
		fputs("<exit9_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit9_y[i]);
		fputs(data, file);
		fputs("</exit9_y>\n", file);
        
		fputs("<drop_point>", file);
        sprintf(data, "%d", h_navmaps_static->drop_point[i]);
		fputs(data, file);
		fputs("</drop_point>\n", file);
        
		fputs("<sandbag_capacity>", file);
        sprintf(data, "%d", h_navmaps_static->sandbag_capacity[i]);
		fputs(data, file);
		fputs("</sandbag_capacity>\n", file);
        
		fputs("<nm_rough>", file);
        sprintf(data, "%f", h_navmaps_static->nm_rough[i]);
		fputs(data, file);
		fputs("</nm_rough>\n", file);
        
		fputs("<evac_counter>", file);
        sprintf(data, "%d", h_navmaps_static->evac_counter[i]);
		fputs(data, file);
		fputs("</evac_counter>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void initEnvVars()
{
PROFILE_SCOPED_RANGE("initEnvVars");

    int t_sandbag_layers = (int)1;
    set_sandbag_layers(&t_sandbag_layers);
}

void readInitialStates(char* inputpath, xmachine_memory_FloodCell_list* h_FloodCells, int* h_xmachine_memory_FloodCell_count,xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count,xmachine_memory_navmap_list* h_navmaps, int* h_xmachine_memory_navmap_count)
{
    PROFILE_SCOPED_RANGE("readInitialStates");

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	const int bufferSize = 10000;
	char buffer[bufferSize];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_xagent, in_name, in_comment;
    int in_FloodCell_inDomain;
    int in_FloodCell_x;
    int in_FloodCell_y;
    int in_FloodCell_z0;
    int in_FloodCell_h;
    int in_FloodCell_qx;
    int in_FloodCell_qy;
    int in_FloodCell_timeStep;
    int in_FloodCell_minh_loc;
    int in_FloodCell_hFace_E;
    int in_FloodCell_etFace_E;
    int in_FloodCell_qxFace_E;
    int in_FloodCell_qyFace_E;
    int in_FloodCell_hFace_W;
    int in_FloodCell_etFace_W;
    int in_FloodCell_qxFace_W;
    int in_FloodCell_qyFace_W;
    int in_FloodCell_hFace_N;
    int in_FloodCell_etFace_N;
    int in_FloodCell_qxFace_N;
    int in_FloodCell_qyFace_N;
    int in_FloodCell_hFace_S;
    int in_FloodCell_etFace_S;
    int in_FloodCell_qxFace_S;
    int in_FloodCell_qyFace_S;
    int in_FloodCell_nm_rough;
    int in_agent_x;
    int in_agent_y;
    int in_agent_velx;
    int in_agent_vely;
    int in_agent_steer_x;
    int in_agent_steer_y;
    int in_agent_height;
    int in_agent_exit_no;
    int in_agent_speed;
    int in_agent_lod;
    int in_agent_animate;
    int in_agent_animate_dir;
    int in_agent_HR_state;
    int in_agent_hero_status;
    int in_agent_pickup_time;
    int in_agent_drop_time;
    int in_agent_carry_sandbag;
    int in_agent_HR;
    int in_agent_dt_ped;
    int in_agent_d_water;
    int in_agent_v_water;
    int in_agent_body_height;
    int in_agent_body_mass;
    int in_agent_gender;
    int in_agent_stability_state;
    int in_agent_motion_speed;
    int in_agent_age;
    int in_agent_excitement_speed;
    int in_agent_dir_times;
    int in_agent_rejected_exit1;
    int in_agent_rejected_exit2;
    int in_agent_rejected_exit3;
    int in_agent_rejected_exit4;
    int in_agent_rejected_exit5;
    int in_navmap_x;
    int in_navmap_y;
    int in_navmap_z0;
    int in_navmap_h;
    int in_navmap_qx;
    int in_navmap_qy;
    int in_navmap_exit_no;
    int in_navmap_height;
    int in_navmap_collision_x;
    int in_navmap_collision_y;
    int in_navmap_exit0_x;
    int in_navmap_exit0_y;
    int in_navmap_exit1_x;
    int in_navmap_exit1_y;
    int in_navmap_exit2_x;
    int in_navmap_exit2_y;
    int in_navmap_exit3_x;
    int in_navmap_exit3_y;
    int in_navmap_exit4_x;
    int in_navmap_exit4_y;
    int in_navmap_exit5_x;
    int in_navmap_exit5_y;
    int in_navmap_exit6_x;
    int in_navmap_exit6_y;
    int in_navmap_exit7_x;
    int in_navmap_exit7_y;
    int in_navmap_exit8_x;
    int in_navmap_exit8_y;
    int in_navmap_exit9_x;
    int in_navmap_exit9_y;
    int in_navmap_drop_point;
    int in_navmap_sandbag_capacity;
    int in_navmap_nm_rough;
    int in_navmap_evac_counter;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_outputting_time;
    
    int in_env_outputting_time_interval;
    
    int in_env_xmin;
    
    int in_env_xmax;
    
    int in_env_ymin;
    
    int in_env_ymax;
    
    int in_env_dt_ped;
    
    int in_env_dt_flood;
    
    int in_env_dt;
    
    int in_env_auto_dt_on;
    
    int in_env_body_as_obstacle_on;
    
    int in_env_ped_roughness_effect_on;
    
    int in_env_body_height;
    
    int in_env_init_speed;
    
    int in_env_brisk_speed;
    
    int in_env_sim_time;
    
    int in_env_DXL;
    
    int in_env_DYL;
    
    int in_env_inflow_start_time;
    
    int in_env_inflow_peak_time;
    
    int in_env_inflow_end_time;
    
    int in_env_inflow_initial_discharge;
    
    int in_env_inflow_peak_discharge;
    
    int in_env_inflow_end_discharge;
    
    int in_env_INFLOW_BOUNDARY;
    
    int in_env_BOUNDARY_EAST_STATUS;
    
    int in_env_BOUNDARY_WEST_STATUS;
    
    int in_env_BOUNDARY_NORTH_STATUS;
    
    int in_env_BOUNDARY_SOUTH_STATUS;
    
    int in_env_x1_boundary;
    
    int in_env_x2_boundary;
    
    int in_env_y1_boundary;
    
    int in_env_y2_boundary;
    
    int in_env_init_depth_boundary;
    
    int in_env_evacuation_on;
    
    int in_env_walking_speed_reduction_in_water_on;
    
    int in_env_freeze_while_instable_on;
    
    int in_env_evacuation_end_time;
    
    int in_env_evacuation_start_time;
    
    int in_env_emergency_exit_number;
    
    int in_env_emer_alarm;
    
    int in_env_HR;
    
    int in_env_max_at_highest_risk;
    
    int in_env_max_at_low_risk;
    
    int in_env_max_at_medium_risk;
    
    int in_env_max_at_high_risk;
    
    int in_env_max_velocity;
    
    int in_env_max_depth;
    
    int in_env_count_population;
    
    int in_env_count_heros;
    
    int in_env_initial_population;
    
    int in_env_evacuated_population;
    
    int in_env_hero_percentage;
    
    int in_env_hero_population;
    
    int in_env_sandbagging_on;
    
    int in_env_sandbagging_start_time;
    
    int in_env_sandbagging_end_time;
    
    int in_env_sandbag_length;
    
    int in_env_sandbag_height;
    
    int in_env_sandbag_width;
    
    int in_env_extended_length;
    
    int in_env_sandbag_layers;
    
    int in_env_update_stopper;
    
    int in_env_dike_length;
    
    int in_env_dike_height;
    
    int in_env_dike_width;
    
    int in_env_fill_cap;
    
    int in_env_pickup_point;
    
    int in_env_drop_point;
    
    int in_env_pickup_duration;
    
    int in_env_drop_duration;
    
    int in_env_EMMISION_RATE_EXIT1;
    
    int in_env_EMMISION_RATE_EXIT2;
    
    int in_env_EMMISION_RATE_EXIT3;
    
    int in_env_EMMISION_RATE_EXIT4;
    
    int in_env_EMMISION_RATE_EXIT5;
    
    int in_env_EMMISION_RATE_EXIT6;
    
    int in_env_EMMISION_RATE_EXIT7;
    
    int in_env_EMMISION_RATE_EXIT8;
    
    int in_env_EMMISION_RATE_EXIT9;
    
    int in_env_EMMISION_RATE_EXIT10;
    
    int in_env_EXIT1_PROBABILITY;
    
    int in_env_EXIT2_PROBABILITY;
    
    int in_env_EXIT3_PROBABILITY;
    
    int in_env_EXIT4_PROBABILITY;
    
    int in_env_EXIT5_PROBABILITY;
    
    int in_env_EXIT6_PROBABILITY;
    
    int in_env_EXIT7_PROBABILITY;
    
    int in_env_EXIT8_PROBABILITY;
    
    int in_env_EXIT9_PROBABILITY;
    
    int in_env_EXIT10_PROBABILITY;
    
    int in_env_EXIT1_STATE;
    
    int in_env_EXIT2_STATE;
    
    int in_env_EXIT3_STATE;
    
    int in_env_EXIT4_STATE;
    
    int in_env_EXIT5_STATE;
    
    int in_env_EXIT6_STATE;
    
    int in_env_EXIT7_STATE;
    
    int in_env_EXIT8_STATE;
    
    int in_env_EXIT9_STATE;
    
    int in_env_EXIT10_STATE;
    
    int in_env_EXIT1_CELL_COUNT;
    
    int in_env_EXIT2_CELL_COUNT;
    
    int in_env_EXIT3_CELL_COUNT;
    
    int in_env_EXIT4_CELL_COUNT;
    
    int in_env_EXIT5_CELL_COUNT;
    
    int in_env_EXIT6_CELL_COUNT;
    
    int in_env_EXIT7_CELL_COUNT;
    
    int in_env_EXIT8_CELL_COUNT;
    
    int in_env_EXIT9_CELL_COUNT;
    
    int in_env_EXIT10_CELL_COUNT;
    
    int in_env_TIME_SCALER;
    
    int in_env_STEER_WEIGHT;
    
    int in_env_AVOID_WEIGHT;
    
    int in_env_COLLISION_WEIGHT;
    
    int in_env_GOAL_WEIGHT;
    
    int in_env_PedHeight_60_110_probability;
    
    int in_env_PedHeight_110_140_probability;
    
    int in_env_PedHeight_140_163_probability;
    
    int in_env_PedHeight_163_170_probability;
    
    int in_env_PedHeight_170_186_probability;
    
    int in_env_PedHeight_186_194_probability;
    
    int in_env_PedHeight_194_210_probability;
    
    int in_env_PedAge_10_17_probability;
    
    int in_env_PedAge_18_29_probability;
    
    int in_env_PedAge_30_39_probability;
    
    int in_env_PedAge_40_49_probability;
    
    int in_env_PedAge_50_59_probability;
    
    int in_env_PedAge_60_69_probability;
    
    int in_env_PedAge_70_79_probability;
    
    int in_env_excluded_age_probability;
    
    int in_env_gender_female_probability;
    
    int in_env_gender_male_probability;
    
    int in_env_SCALE_FACTOR;
    
    int in_env_I_SCALER;
    
    int in_env_MIN_DISTANCE;
    
    int in_env_excitement_on;
    
    int in_env_walk_run_switch;
    
    int in_env_preoccupying_on;
    
    int in_env_poly_hydrograph_on;
    
    int in_env_stop_emission_on;
    
    int in_env_goto_emergency_exit_on;
    
    int in_env_escape_route_finder_on;
    
    int in_env_dir_times;
    
    int in_env_no_return_on;
    
    int in_env_wdepth_perc_thresh;
    
    int in_env_follow_popular_exit_on;
    
    int in_env_popular_exit;
    
	/* set agent count to zero */
	*h_xmachine_memory_FloodCell_count = 0;
	*h_xmachine_memory_agent_count = 0;
	*h_xmachine_memory_navmap_count = 0;
	
	/* Variables for initial state data */
	int FloodCell_inDomain;
	int FloodCell_x;
	int FloodCell_y;
	double FloodCell_z0;
	double FloodCell_h;
	double FloodCell_qx;
	double FloodCell_qy;
	double FloodCell_timeStep;
	double FloodCell_minh_loc;
	double FloodCell_hFace_E;
	double FloodCell_etFace_E;
	double FloodCell_qxFace_E;
	double FloodCell_qyFace_E;
	double FloodCell_hFace_W;
	double FloodCell_etFace_W;
	double FloodCell_qxFace_W;
	double FloodCell_qyFace_W;
	double FloodCell_hFace_N;
	double FloodCell_etFace_N;
	double FloodCell_qxFace_N;
	double FloodCell_qyFace_N;
	double FloodCell_hFace_S;
	double FloodCell_etFace_S;
	double FloodCell_qxFace_S;
	double FloodCell_qyFace_S;
	double FloodCell_nm_rough;
	float agent_x;
	float agent_y;
	float agent_velx;
	float agent_vely;
	float agent_steer_x;
	float agent_steer_y;
	float agent_height;
	int agent_exit_no;
	float agent_speed;
	int agent_lod;
	float agent_animate;
	int agent_animate_dir;
	int agent_HR_state;
	int agent_hero_status;
	double agent_pickup_time;
	double agent_drop_time;
	int agent_carry_sandbag;
	double agent_HR;
	float agent_dt_ped;
	float agent_d_water;
	float agent_v_water;
	float agent_body_height;
	float agent_body_mass;
	int agent_gender;
	int agent_stability_state;
	float agent_motion_speed;
	int agent_age;
	float agent_excitement_speed;
	int agent_dir_times;
	int agent_rejected_exit1;
	int agent_rejected_exit2;
	int agent_rejected_exit3;
	int agent_rejected_exit4;
	int agent_rejected_exit5;
	int navmap_x;
	int navmap_y;
	double navmap_z0;
	double navmap_h;
	double navmap_qx;
	double navmap_qy;
	int navmap_exit_no;
	float navmap_height;
	float navmap_collision_x;
	float navmap_collision_y;
	float navmap_exit0_x;
	float navmap_exit0_y;
	float navmap_exit1_x;
	float navmap_exit1_y;
	float navmap_exit2_x;
	float navmap_exit2_y;
	float navmap_exit3_x;
	float navmap_exit3_y;
	float navmap_exit4_x;
	float navmap_exit4_y;
	float navmap_exit5_x;
	float navmap_exit5_y;
	float navmap_exit6_x;
	float navmap_exit6_y;
	float navmap_exit7_x;
	float navmap_exit7_y;
	float navmap_exit8_x;
	float navmap_exit8_y;
	float navmap_exit9_x;
	float navmap_exit9_y;
	int navmap_drop_point;
	int navmap_sandbag_capacity;
	double navmap_nm_rough;
	int navmap_evac_counter;

    /* Variables for environment variables */
    double env_outputting_time;
    double env_outputting_time_interval;
    double env_xmin;
    double env_xmax;
    double env_ymin;
    double env_ymax;
    double env_dt_ped;
    double env_dt_flood;
    double env_dt;
    int env_auto_dt_on;
    int env_body_as_obstacle_on;
    int env_ped_roughness_effect_on;
    float env_body_height;
    float env_init_speed;
    float env_brisk_speed;
    double env_sim_time;
    double env_DXL;
    double env_DYL;
    double env_inflow_start_time;
    double env_inflow_peak_time;
    double env_inflow_end_time;
    double env_inflow_initial_discharge;
    double env_inflow_peak_discharge;
    double env_inflow_end_discharge;
    int env_INFLOW_BOUNDARY;
    int env_BOUNDARY_EAST_STATUS;
    int env_BOUNDARY_WEST_STATUS;
    int env_BOUNDARY_NORTH_STATUS;
    int env_BOUNDARY_SOUTH_STATUS;
    double env_x1_boundary;
    double env_x2_boundary;
    double env_y1_boundary;
    double env_y2_boundary;
    double env_init_depth_boundary;
    int env_evacuation_on;
    int env_walking_speed_reduction_in_water_on;
    int env_freeze_while_instable_on;
    double env_evacuation_end_time;
    double env_evacuation_start_time;
    int env_emergency_exit_number;
    int env_emer_alarm;
    double env_HR;
    int env_max_at_highest_risk;
    int env_max_at_low_risk;
    int env_max_at_medium_risk;
    int env_max_at_high_risk;
    double env_max_velocity;
    double env_max_depth;
    int env_count_population;
    int env_count_heros;
    int env_initial_population;
    int env_evacuated_population;
    float env_hero_percentage;
    int env_hero_population;
    int env_sandbagging_on;
    double env_sandbagging_start_time;
    double env_sandbagging_end_time;
    float env_sandbag_length;
    float env_sandbag_height;
    float env_sandbag_width;
    float env_extended_length;
    int env_sandbag_layers;
    int env_update_stopper;
    float env_dike_length;
    float env_dike_height;
    float env_dike_width;
    int env_fill_cap;
    int env_pickup_point;
    int env_drop_point;
    float env_pickup_duration;
    float env_drop_duration;
    float env_EMMISION_RATE_EXIT1;
    float env_EMMISION_RATE_EXIT2;
    float env_EMMISION_RATE_EXIT3;
    float env_EMMISION_RATE_EXIT4;
    float env_EMMISION_RATE_EXIT5;
    float env_EMMISION_RATE_EXIT6;
    float env_EMMISION_RATE_EXIT7;
    float env_EMMISION_RATE_EXIT8;
    float env_EMMISION_RATE_EXIT9;
    float env_EMMISION_RATE_EXIT10;
    int env_EXIT1_PROBABILITY;
    int env_EXIT2_PROBABILITY;
    int env_EXIT3_PROBABILITY;
    int env_EXIT4_PROBABILITY;
    int env_EXIT5_PROBABILITY;
    int env_EXIT6_PROBABILITY;
    int env_EXIT7_PROBABILITY;
    int env_EXIT8_PROBABILITY;
    int env_EXIT9_PROBABILITY;
    int env_EXIT10_PROBABILITY;
    int env_EXIT1_STATE;
    int env_EXIT2_STATE;
    int env_EXIT3_STATE;
    int env_EXIT4_STATE;
    int env_EXIT5_STATE;
    int env_EXIT6_STATE;
    int env_EXIT7_STATE;
    int env_EXIT8_STATE;
    int env_EXIT9_STATE;
    int env_EXIT10_STATE;
    int env_EXIT1_CELL_COUNT;
    int env_EXIT2_CELL_COUNT;
    int env_EXIT3_CELL_COUNT;
    int env_EXIT4_CELL_COUNT;
    int env_EXIT5_CELL_COUNT;
    int env_EXIT6_CELL_COUNT;
    int env_EXIT7_CELL_COUNT;
    int env_EXIT8_CELL_COUNT;
    int env_EXIT9_CELL_COUNT;
    int env_EXIT10_CELL_COUNT;
    float env_TIME_SCALER;
    float env_STEER_WEIGHT;
    float env_AVOID_WEIGHT;
    float env_COLLISION_WEIGHT;
    float env_GOAL_WEIGHT;
    int env_PedHeight_60_110_probability;
    int env_PedHeight_110_140_probability;
    int env_PedHeight_140_163_probability;
    int env_PedHeight_163_170_probability;
    int env_PedHeight_170_186_probability;
    int env_PedHeight_186_194_probability;
    int env_PedHeight_194_210_probability;
    int env_PedAge_10_17_probability;
    int env_PedAge_18_29_probability;
    int env_PedAge_30_39_probability;
    int env_PedAge_40_49_probability;
    int env_PedAge_50_59_probability;
    int env_PedAge_60_69_probability;
    int env_PedAge_70_79_probability;
    int env_excluded_age_probability;
    int env_gender_female_probability;
    int env_gender_male_probability;
    float env_SCALE_FACTOR;
    float env_I_SCALER;
    float env_MIN_DISTANCE;
    int env_excitement_on;
    int env_walk_run_switch;
    int env_preoccupying_on;
    int env_poly_hydrograph_on;
    int env_stop_emission_on;
    int env_goto_emergency_exit_on;
    int env_escape_route_finder_on;
    int env_dir_times;
    int env_no_return_on;
    float env_wdepth_perc_thresh;
    int env_follow_popular_exit_on;
    int env_popular_exit;
    


	/* Initialise variables */
    initEnvVars();
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
    in_comment = 0;
	in_tag = 0;
	in_itno = 0;
    in_env = 0;
    in_xagent = 0;
	in_name = 0;
	in_FloodCell_inDomain = 0;
	in_FloodCell_x = 0;
	in_FloodCell_y = 0;
	in_FloodCell_z0 = 0;
	in_FloodCell_h = 0;
	in_FloodCell_qx = 0;
	in_FloodCell_qy = 0;
	in_FloodCell_timeStep = 0;
	in_FloodCell_minh_loc = 0;
	in_FloodCell_hFace_E = 0;
	in_FloodCell_etFace_E = 0;
	in_FloodCell_qxFace_E = 0;
	in_FloodCell_qyFace_E = 0;
	in_FloodCell_hFace_W = 0;
	in_FloodCell_etFace_W = 0;
	in_FloodCell_qxFace_W = 0;
	in_FloodCell_qyFace_W = 0;
	in_FloodCell_hFace_N = 0;
	in_FloodCell_etFace_N = 0;
	in_FloodCell_qxFace_N = 0;
	in_FloodCell_qyFace_N = 0;
	in_FloodCell_hFace_S = 0;
	in_FloodCell_etFace_S = 0;
	in_FloodCell_qxFace_S = 0;
	in_FloodCell_qyFace_S = 0;
	in_FloodCell_nm_rough = 0;
	in_agent_x = 0;
	in_agent_y = 0;
	in_agent_velx = 0;
	in_agent_vely = 0;
	in_agent_steer_x = 0;
	in_agent_steer_y = 0;
	in_agent_height = 0;
	in_agent_exit_no = 0;
	in_agent_speed = 0;
	in_agent_lod = 0;
	in_agent_animate = 0;
	in_agent_animate_dir = 0;
	in_agent_HR_state = 0;
	in_agent_hero_status = 0;
	in_agent_pickup_time = 0;
	in_agent_drop_time = 0;
	in_agent_carry_sandbag = 0;
	in_agent_HR = 0;
	in_agent_dt_ped = 0;
	in_agent_d_water = 0;
	in_agent_v_water = 0;
	in_agent_body_height = 0;
	in_agent_body_mass = 0;
	in_agent_gender = 0;
	in_agent_stability_state = 0;
	in_agent_motion_speed = 0;
	in_agent_age = 0;
	in_agent_excitement_speed = 0;
	in_agent_dir_times = 0;
	in_agent_rejected_exit1 = 0;
	in_agent_rejected_exit2 = 0;
	in_agent_rejected_exit3 = 0;
	in_agent_rejected_exit4 = 0;
	in_agent_rejected_exit5 = 0;
	in_navmap_x = 0;
	in_navmap_y = 0;
	in_navmap_z0 = 0;
	in_navmap_h = 0;
	in_navmap_qx = 0;
	in_navmap_qy = 0;
	in_navmap_exit_no = 0;
	in_navmap_height = 0;
	in_navmap_collision_x = 0;
	in_navmap_collision_y = 0;
	in_navmap_exit0_x = 0;
	in_navmap_exit0_y = 0;
	in_navmap_exit1_x = 0;
	in_navmap_exit1_y = 0;
	in_navmap_exit2_x = 0;
	in_navmap_exit2_y = 0;
	in_navmap_exit3_x = 0;
	in_navmap_exit3_y = 0;
	in_navmap_exit4_x = 0;
	in_navmap_exit4_y = 0;
	in_navmap_exit5_x = 0;
	in_navmap_exit5_y = 0;
	in_navmap_exit6_x = 0;
	in_navmap_exit6_y = 0;
	in_navmap_exit7_x = 0;
	in_navmap_exit7_y = 0;
	in_navmap_exit8_x = 0;
	in_navmap_exit8_y = 0;
	in_navmap_exit9_x = 0;
	in_navmap_exit9_y = 0;
	in_navmap_drop_point = 0;
	in_navmap_sandbag_capacity = 0;
	in_navmap_nm_rough = 0;
	in_navmap_evac_counter = 0;
    in_env_outputting_time = 0;
    in_env_outputting_time_interval = 0;
    in_env_xmin = 0;
    in_env_xmax = 0;
    in_env_ymin = 0;
    in_env_ymax = 0;
    in_env_dt_ped = 0;
    in_env_dt_flood = 0;
    in_env_dt = 0;
    in_env_auto_dt_on = 0;
    in_env_body_as_obstacle_on = 0;
    in_env_ped_roughness_effect_on = 0;
    in_env_body_height = 0;
    in_env_init_speed = 0;
    in_env_brisk_speed = 0;
    in_env_sim_time = 0;
    in_env_DXL = 0;
    in_env_DYL = 0;
    in_env_inflow_start_time = 0;
    in_env_inflow_peak_time = 0;
    in_env_inflow_end_time = 0;
    in_env_inflow_initial_discharge = 0;
    in_env_inflow_peak_discharge = 0;
    in_env_inflow_end_discharge = 0;
    in_env_INFLOW_BOUNDARY = 0;
    in_env_BOUNDARY_EAST_STATUS = 0;
    in_env_BOUNDARY_WEST_STATUS = 0;
    in_env_BOUNDARY_NORTH_STATUS = 0;
    in_env_BOUNDARY_SOUTH_STATUS = 0;
    in_env_x1_boundary = 0;
    in_env_x2_boundary = 0;
    in_env_y1_boundary = 0;
    in_env_y2_boundary = 0;
    in_env_init_depth_boundary = 0;
    in_env_evacuation_on = 0;
    in_env_walking_speed_reduction_in_water_on = 0;
    in_env_freeze_while_instable_on = 0;
    in_env_evacuation_end_time = 0;
    in_env_evacuation_start_time = 0;
    in_env_emergency_exit_number = 0;
    in_env_emer_alarm = 0;
    in_env_HR = 0;
    in_env_max_at_highest_risk = 0;
    in_env_max_at_low_risk = 0;
    in_env_max_at_medium_risk = 0;
    in_env_max_at_high_risk = 0;
    in_env_max_velocity = 0;
    in_env_max_depth = 0;
    in_env_count_population = 0;
    in_env_count_heros = 0;
    in_env_initial_population = 0;
    in_env_evacuated_population = 0;
    in_env_hero_percentage = 0;
    in_env_hero_population = 0;
    in_env_sandbagging_on = 0;
    in_env_sandbagging_start_time = 0;
    in_env_sandbagging_end_time = 0;
    in_env_sandbag_length = 0;
    in_env_sandbag_height = 0;
    in_env_sandbag_width = 0;
    in_env_extended_length = 0;
    in_env_sandbag_layers = 0;
    in_env_update_stopper = 0;
    in_env_dike_length = 0;
    in_env_dike_height = 0;
    in_env_dike_width = 0;
    in_env_fill_cap = 0;
    in_env_pickup_point = 0;
    in_env_drop_point = 0;
    in_env_pickup_duration = 0;
    in_env_drop_duration = 0;
    in_env_EMMISION_RATE_EXIT1 = 0;
    in_env_EMMISION_RATE_EXIT2 = 0;
    in_env_EMMISION_RATE_EXIT3 = 0;
    in_env_EMMISION_RATE_EXIT4 = 0;
    in_env_EMMISION_RATE_EXIT5 = 0;
    in_env_EMMISION_RATE_EXIT6 = 0;
    in_env_EMMISION_RATE_EXIT7 = 0;
    in_env_EMMISION_RATE_EXIT8 = 0;
    in_env_EMMISION_RATE_EXIT9 = 0;
    in_env_EMMISION_RATE_EXIT10 = 0;
    in_env_EXIT1_PROBABILITY = 0;
    in_env_EXIT2_PROBABILITY = 0;
    in_env_EXIT3_PROBABILITY = 0;
    in_env_EXIT4_PROBABILITY = 0;
    in_env_EXIT5_PROBABILITY = 0;
    in_env_EXIT6_PROBABILITY = 0;
    in_env_EXIT7_PROBABILITY = 0;
    in_env_EXIT8_PROBABILITY = 0;
    in_env_EXIT9_PROBABILITY = 0;
    in_env_EXIT10_PROBABILITY = 0;
    in_env_EXIT1_STATE = 0;
    in_env_EXIT2_STATE = 0;
    in_env_EXIT3_STATE = 0;
    in_env_EXIT4_STATE = 0;
    in_env_EXIT5_STATE = 0;
    in_env_EXIT6_STATE = 0;
    in_env_EXIT7_STATE = 0;
    in_env_EXIT8_STATE = 0;
    in_env_EXIT9_STATE = 0;
    in_env_EXIT10_STATE = 0;
    in_env_EXIT1_CELL_COUNT = 0;
    in_env_EXIT2_CELL_COUNT = 0;
    in_env_EXIT3_CELL_COUNT = 0;
    in_env_EXIT4_CELL_COUNT = 0;
    in_env_EXIT5_CELL_COUNT = 0;
    in_env_EXIT6_CELL_COUNT = 0;
    in_env_EXIT7_CELL_COUNT = 0;
    in_env_EXIT8_CELL_COUNT = 0;
    in_env_EXIT9_CELL_COUNT = 0;
    in_env_EXIT10_CELL_COUNT = 0;
    in_env_TIME_SCALER = 0;
    in_env_STEER_WEIGHT = 0;
    in_env_AVOID_WEIGHT = 0;
    in_env_COLLISION_WEIGHT = 0;
    in_env_GOAL_WEIGHT = 0;
    in_env_PedHeight_60_110_probability = 0;
    in_env_PedHeight_110_140_probability = 0;
    in_env_PedHeight_140_163_probability = 0;
    in_env_PedHeight_163_170_probability = 0;
    in_env_PedHeight_170_186_probability = 0;
    in_env_PedHeight_186_194_probability = 0;
    in_env_PedHeight_194_210_probability = 0;
    in_env_PedAge_10_17_probability = 0;
    in_env_PedAge_18_29_probability = 0;
    in_env_PedAge_30_39_probability = 0;
    in_env_PedAge_40_49_probability = 0;
    in_env_PedAge_50_59_probability = 0;
    in_env_PedAge_60_69_probability = 0;
    in_env_PedAge_70_79_probability = 0;
    in_env_excluded_age_probability = 0;
    in_env_gender_female_probability = 0;
    in_env_gender_male_probability = 0;
    in_env_SCALE_FACTOR = 0;
    in_env_I_SCALER = 0;
    in_env_MIN_DISTANCE = 0;
    in_env_excitement_on = 0;
    in_env_walk_run_switch = 0;
    in_env_preoccupying_on = 0;
    in_env_poly_hydrograph_on = 0;
    in_env_stop_emission_on = 0;
    in_env_goto_emergency_exit_on = 0;
    in_env_escape_route_finder_on = 0;
    in_env_dir_times = 0;
    in_env_no_return_on = 0;
    in_env_wdepth_perc_thresh = 0;
    in_env_follow_popular_exit_on = 0;
    in_env_popular_exit = 0;
	//set all FloodCell values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_FloodCell_MAX; k++)
	{	
		h_FloodCells->inDomain[k] = 0;
		h_FloodCells->x[k] = 0;
		h_FloodCells->y[k] = 0;
		h_FloodCells->z0[k] = 0;
		h_FloodCells->h[k] = 0;
		h_FloodCells->qx[k] = 0;
		h_FloodCells->qy[k] = 0;
		h_FloodCells->timeStep[k] = 0;
		h_FloodCells->minh_loc[k] = 0;
		h_FloodCells->hFace_E[k] = 0;
		h_FloodCells->etFace_E[k] = 0;
		h_FloodCells->qxFace_E[k] = 0;
		h_FloodCells->qyFace_E[k] = 0;
		h_FloodCells->hFace_W[k] = 0;
		h_FloodCells->etFace_W[k] = 0;
		h_FloodCells->qxFace_W[k] = 0;
		h_FloodCells->qyFace_W[k] = 0;
		h_FloodCells->hFace_N[k] = 0;
		h_FloodCells->etFace_N[k] = 0;
		h_FloodCells->qxFace_N[k] = 0;
		h_FloodCells->qyFace_N[k] = 0;
		h_FloodCells->hFace_S[k] = 0;
		h_FloodCells->etFace_S[k] = 0;
		h_FloodCells->qxFace_S[k] = 0;
		h_FloodCells->qyFace_S[k] = 0;
		h_FloodCells->nm_rough[k] = 0.01100f;
	}
	
	//set all agent values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_agent_MAX; k++)
	{	
		h_agents->x[k] = 0;
		h_agents->y[k] = 0;
		h_agents->velx[k] = 0;
		h_agents->vely[k] = 0;
		h_agents->steer_x[k] = 0;
		h_agents->steer_y[k] = 0;
		h_agents->height[k] = 0;
		h_agents->exit_no[k] = 0;
		h_agents->speed[k] = 0;
		h_agents->lod[k] = 0;
		h_agents->animate[k] = 0;
		h_agents->animate_dir[k] = 0;
		h_agents->HR_state[k] = 0;
		h_agents->hero_status[k] = 0;
		h_agents->pickup_time[k] = 0;
		h_agents->drop_time[k] = 0;
		h_agents->carry_sandbag[k] = 0;
		h_agents->HR[k] = 0;
		h_agents->dt_ped[k] = 0;
		h_agents->d_water[k] = 0;
		h_agents->v_water[k] = 0;
		h_agents->body_height[k] = 0;
		h_agents->body_mass[k] = 0;
		h_agents->gender[k] = 0;
		h_agents->stability_state[k] = 0;
		h_agents->motion_speed[k] = 0;
		h_agents->age[k] = 0;
		h_agents->excitement_speed[k] = 0;
		h_agents->dir_times[k] = 0;
		h_agents->rejected_exit1[k] = 0;
		h_agents->rejected_exit2[k] = 0;
		h_agents->rejected_exit3[k] = 0;
		h_agents->rejected_exit4[k] = 0;
		h_agents->rejected_exit5[k] = 0;
	}
	
	//set all navmap values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_navmap_MAX; k++)
	{	
		h_navmaps->x[k] = 0;
		h_navmaps->y[k] = 0;
		h_navmaps->z0[k] = 0;
		h_navmaps->h[k] = 0;
		h_navmaps->qx[k] = 0;
		h_navmaps->qy[k] = 0;
		h_navmaps->exit_no[k] = 0;
		h_navmaps->height[k] = 0;
		h_navmaps->collision_x[k] = 0;
		h_navmaps->collision_y[k] = 0;
		h_navmaps->exit0_x[k] = 0;
		h_navmaps->exit0_y[k] = 0;
		h_navmaps->exit1_x[k] = 0;
		h_navmaps->exit1_y[k] = 0;
		h_navmaps->exit2_x[k] = 0;
		h_navmaps->exit2_y[k] = 0;
		h_navmaps->exit3_x[k] = 0;
		h_navmaps->exit3_y[k] = 0;
		h_navmaps->exit4_x[k] = 0;
		h_navmaps->exit4_y[k] = 0;
		h_navmaps->exit5_x[k] = 0;
		h_navmaps->exit5_y[k] = 0;
		h_navmaps->exit6_x[k] = 0;
		h_navmaps->exit6_y[k] = 0;
		h_navmaps->exit7_x[k] = 0;
		h_navmaps->exit7_y[k] = 0;
		h_navmaps->exit8_x[k] = 0;
		h_navmaps->exit8_y[k] = 0;
		h_navmaps->exit9_x[k] = 0;
		h_navmaps->exit9_y[k] = 0;
		h_navmaps->drop_point[k] = 0;
		h_navmaps->sandbag_capacity[k] = 0;
		h_navmaps->nm_rough[k] = 0;
		h_navmaps->evac_counter[k] = 0;
	}
	

	/* Default variables for memory */
    FloodCell_inDomain = 0;
    FloodCell_x = 0;
    FloodCell_y = 0;
    FloodCell_z0 = 0;
    FloodCell_h = 0;
    FloodCell_qx = 0;
    FloodCell_qy = 0;
    FloodCell_timeStep = 0;
    FloodCell_minh_loc = 0;
    FloodCell_hFace_E = 0;
    FloodCell_etFace_E = 0;
    FloodCell_qxFace_E = 0;
    FloodCell_qyFace_E = 0;
    FloodCell_hFace_W = 0;
    FloodCell_etFace_W = 0;
    FloodCell_qxFace_W = 0;
    FloodCell_qyFace_W = 0;
    FloodCell_hFace_N = 0;
    FloodCell_etFace_N = 0;
    FloodCell_qxFace_N = 0;
    FloodCell_qyFace_N = 0;
    FloodCell_hFace_S = 0;
    FloodCell_etFace_S = 0;
    FloodCell_qxFace_S = 0;
    FloodCell_qyFace_S = 0;
    FloodCell_nm_rough = 0.01100f;
    agent_x = 0;
    agent_y = 0;
    agent_velx = 0;
    agent_vely = 0;
    agent_steer_x = 0;
    agent_steer_y = 0;
    agent_height = 0;
    agent_exit_no = 0;
    agent_speed = 0;
    agent_lod = 0;
    agent_animate = 0;
    agent_animate_dir = 0;
    agent_HR_state = 0;
    agent_hero_status = 0;
    agent_pickup_time = 0;
    agent_drop_time = 0;
    agent_carry_sandbag = 0;
    agent_HR = 0;
    agent_dt_ped = 0;
    agent_d_water = 0;
    agent_v_water = 0;
    agent_body_height = 0;
    agent_body_mass = 0;
    agent_gender = 0;
    agent_stability_state = 0;
    agent_motion_speed = 0;
    agent_age = 0;
    agent_excitement_speed = 0;
    agent_dir_times = 0;
    agent_rejected_exit1 = 0;
    agent_rejected_exit2 = 0;
    agent_rejected_exit3 = 0;
    agent_rejected_exit4 = 0;
    agent_rejected_exit5 = 0;
    navmap_x = 0;
    navmap_y = 0;
    navmap_z0 = 0;
    navmap_h = 0;
    navmap_qx = 0;
    navmap_qy = 0;
    navmap_exit_no = 0;
    navmap_height = 0;
    navmap_collision_x = 0;
    navmap_collision_y = 0;
    navmap_exit0_x = 0;
    navmap_exit0_y = 0;
    navmap_exit1_x = 0;
    navmap_exit1_y = 0;
    navmap_exit2_x = 0;
    navmap_exit2_y = 0;
    navmap_exit3_x = 0;
    navmap_exit3_y = 0;
    navmap_exit4_x = 0;
    navmap_exit4_y = 0;
    navmap_exit5_x = 0;
    navmap_exit5_y = 0;
    navmap_exit6_x = 0;
    navmap_exit6_y = 0;
    navmap_exit7_x = 0;
    navmap_exit7_y = 0;
    navmap_exit8_x = 0;
    navmap_exit8_y = 0;
    navmap_exit9_x = 0;
    navmap_exit9_y = 0;
    navmap_drop_point = 0;
    navmap_sandbag_capacity = 0;
    navmap_nm_rough = 0;
    navmap_evac_counter = 0;

    /* Default variables for environment variables */
    env_outputting_time = 0;
    env_outputting_time_interval = 0;
    env_xmin = 0;
    env_xmax = 0;
    env_ymin = 0;
    env_ymax = 0;
    env_dt_ped = 0;
    env_dt_flood = 0;
    env_dt = 0;
    env_auto_dt_on = 0;
    env_body_as_obstacle_on = 0;
    env_ped_roughness_effect_on = 0;
    env_body_height = 0;
    env_init_speed = 0;
    env_brisk_speed = 0;
    env_sim_time = 0;
    env_DXL = 0;
    env_DYL = 0;
    env_inflow_start_time = 0;
    env_inflow_peak_time = 0;
    env_inflow_end_time = 0;
    env_inflow_initial_discharge = 0;
    env_inflow_peak_discharge = 0;
    env_inflow_end_discharge = 0;
    env_INFLOW_BOUNDARY = 0;
    env_BOUNDARY_EAST_STATUS = 0;
    env_BOUNDARY_WEST_STATUS = 0;
    env_BOUNDARY_NORTH_STATUS = 0;
    env_BOUNDARY_SOUTH_STATUS = 0;
    env_x1_boundary = 0;
    env_x2_boundary = 0;
    env_y1_boundary = 0;
    env_y2_boundary = 0;
    env_init_depth_boundary = 0;
    env_evacuation_on = 0;
    env_walking_speed_reduction_in_water_on = 0;
    env_freeze_while_instable_on = 0;
    env_evacuation_end_time = 0;
    env_evacuation_start_time = 0;
    env_emergency_exit_number = 0;
    env_emer_alarm = 0;
    env_HR = 0;
    env_max_at_highest_risk = 0;
    env_max_at_low_risk = 0;
    env_max_at_medium_risk = 0;
    env_max_at_high_risk = 0;
    env_max_velocity = 0;
    env_max_depth = 0;
    env_count_population = 0;
    env_count_heros = 0;
    env_initial_population = 0;
    env_evacuated_population = 0;
    env_hero_percentage = 0;
    env_hero_population = 0;
    env_sandbagging_on = 0;
    env_sandbagging_start_time = 0;
    env_sandbagging_end_time = 0;
    env_sandbag_length = 0;
    env_sandbag_height = 0;
    env_sandbag_width = 0;
    env_extended_length = 0;
    env_sandbag_layers = 1;
    env_update_stopper = 0;
    env_dike_length = 0;
    env_dike_height = 0;
    env_dike_width = 0;
    env_fill_cap = 0;
    env_pickup_point = 0;
    env_drop_point = 0;
    env_pickup_duration = 0;
    env_drop_duration = 0;
    env_EMMISION_RATE_EXIT1 = 0;
    env_EMMISION_RATE_EXIT2 = 0;
    env_EMMISION_RATE_EXIT3 = 0;
    env_EMMISION_RATE_EXIT4 = 0;
    env_EMMISION_RATE_EXIT5 = 0;
    env_EMMISION_RATE_EXIT6 = 0;
    env_EMMISION_RATE_EXIT7 = 0;
    env_EMMISION_RATE_EXIT8 = 0;
    env_EMMISION_RATE_EXIT9 = 0;
    env_EMMISION_RATE_EXIT10 = 0;
    env_EXIT1_PROBABILITY = 0;
    env_EXIT2_PROBABILITY = 0;
    env_EXIT3_PROBABILITY = 0;
    env_EXIT4_PROBABILITY = 0;
    env_EXIT5_PROBABILITY = 0;
    env_EXIT6_PROBABILITY = 0;
    env_EXIT7_PROBABILITY = 0;
    env_EXIT8_PROBABILITY = 0;
    env_EXIT9_PROBABILITY = 0;
    env_EXIT10_PROBABILITY = 0;
    env_EXIT1_STATE = 0;
    env_EXIT2_STATE = 0;
    env_EXIT3_STATE = 0;
    env_EXIT4_STATE = 0;
    env_EXIT5_STATE = 0;
    env_EXIT6_STATE = 0;
    env_EXIT7_STATE = 0;
    env_EXIT8_STATE = 0;
    env_EXIT9_STATE = 0;
    env_EXIT10_STATE = 0;
    env_EXIT1_CELL_COUNT = 0;
    env_EXIT2_CELL_COUNT = 0;
    env_EXIT3_CELL_COUNT = 0;
    env_EXIT4_CELL_COUNT = 0;
    env_EXIT5_CELL_COUNT = 0;
    env_EXIT6_CELL_COUNT = 0;
    env_EXIT7_CELL_COUNT = 0;
    env_EXIT8_CELL_COUNT = 0;
    env_EXIT9_CELL_COUNT = 0;
    env_EXIT10_CELL_COUNT = 0;
    env_TIME_SCALER = 0;
    env_STEER_WEIGHT = 0;
    env_AVOID_WEIGHT = 0;
    env_COLLISION_WEIGHT = 0;
    env_GOAL_WEIGHT = 0;
    env_PedHeight_60_110_probability = 0;
    env_PedHeight_110_140_probability = 0;
    env_PedHeight_140_163_probability = 0;
    env_PedHeight_163_170_probability = 0;
    env_PedHeight_170_186_probability = 0;
    env_PedHeight_186_194_probability = 0;
    env_PedHeight_194_210_probability = 0;
    env_PedAge_10_17_probability = 0;
    env_PedAge_18_29_probability = 0;
    env_PedAge_30_39_probability = 0;
    env_PedAge_40_49_probability = 0;
    env_PedAge_50_59_probability = 0;
    env_PedAge_60_69_probability = 0;
    env_PedAge_70_79_probability = 0;
    env_excluded_age_probability = 0;
    env_gender_female_probability = 0;
    env_gender_male_probability = 0;
    env_SCALE_FACTOR = 0;
    env_I_SCALER = 0;
    env_MIN_DISTANCE = 0;
    env_excitement_on = 0;
    env_walk_run_switch = 0;
    env_preoccupying_on = 0;
    env_poly_hydrograph_on = 0;
    env_stop_emission_on = 0;
    env_goto_emergency_exit_on = 0;
    env_escape_route_finder_on = 0;
    env_dir_times = 0;
    env_no_return_on = 0;
    env_wdepth_perc_thresh = 0;
    env_follow_popular_exit_on = 0;
    env_popular_exit = 0;
    
    
    // If no input path was specified, issue a message and return.
    if(inputpath[0] == '\0'){
        printf("No initial states file specified. Using default values.\n");
        return;
    }
    
    // Otherwise an input path was specified, and we have previously checked that it is (was) not a directory. 
    
	// Attempt to open the non directory path as read only.
	file = fopen(inputpath, "r");
    
    // If the file could not be opened, issue a message and return.
    if(file == nullptr)
    {
      printf("Could not open input file %s. Continuing with default values\n", inputpath);
      return;
    }
    // Otherwise we can iterate the file until the end of XML is reached.
    size_t bytesRead = 0;
    i = 0;
	while(reading==1)
	{
        // If I exceeds our buffer size we must abort
        if(i >= bufferSize){
            fprintf(stderr, "Error: XML Parsing failed Tag name or content too long (> %d characters)\n", bufferSize);
            exit(EXIT_FAILURE);
        }

		/* Get the next char from the file */
		c = (char)fgetc(file);

        // Check if we reached the end of the file.
        if(c == EOF){
            // Break out of the loop. This allows for empty files(which may or may not be)
            break;
        }
        // Increment byte counter.
        bytesRead++;

        /*If in a  comment, look for the end of a comment */
        if(in_comment){

            /* Look for an end tag following two (or more) hyphens.
               To support very long comments, we use the minimal amount of buffer we can. 
               If we see a hyphen, store it and increment i (but don't increment i)
               If we see a > check if we have a correct terminating comment
               If we see any other characters, reset i.
            */

            if(c == '-'){
                buffer[i] = c;
                i++;
            } else if(c == '>' && i >= 2){
                in_comment = 0;
                i = 0;
            } else {
                i = 0;
            }

            /*// If we see the end tag, check the preceding two characters for a close comment, if enough characters have been read for -->
            if(c == '>' && i >= 2 && buffer[i-1] == '-' && buffer[i-2] == '-'){
                in_comment = 0;
                buffer[0] = 0;
                i = 0;
            } else {
                // Otherwise just store it in the buffer so we can keep checking for close tags
                buffer[i] = c;
                i++;
            }*/
        }
		/* If the end of a tag */
		else if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;

			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
            if(strcmp(buffer, "environment") == 0) in_env = 1;
            if(strcmp(buffer, "/environment") == 0) in_env = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
            if(strcmp(buffer, "xagent") == 0) in_xagent = 1;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "FloodCell") == 0)
				{
					if (*h_xmachine_memory_FloodCell_count > xmachine_memory_FloodCell_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent FloodCell exceeded whilst reading data\n", xmachine_memory_FloodCell_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_FloodCells->inDomain[*h_xmachine_memory_FloodCell_count] = FloodCell_inDomain;
					h_FloodCells->x[*h_xmachine_memory_FloodCell_count] = FloodCell_x;//Check maximum x value
                    if(agent_maximum.x < FloodCell_x)
                        agent_maximum.x = (float)FloodCell_x;
                    //Check minimum x value
                    if(agent_minimum.x > FloodCell_x)
                        agent_minimum.x = (float)FloodCell_x;
                    
					h_FloodCells->y[*h_xmachine_memory_FloodCell_count] = FloodCell_y;//Check maximum y value
                    if(agent_maximum.y < FloodCell_y)
                        agent_maximum.y = (float)FloodCell_y;
                    //Check minimum y value
                    if(agent_minimum.y > FloodCell_y)
                        agent_minimum.y = (float)FloodCell_y;
                    
					h_FloodCells->z0[*h_xmachine_memory_FloodCell_count] = FloodCell_z0;
					h_FloodCells->h[*h_xmachine_memory_FloodCell_count] = FloodCell_h;
					h_FloodCells->qx[*h_xmachine_memory_FloodCell_count] = FloodCell_qx;
					h_FloodCells->qy[*h_xmachine_memory_FloodCell_count] = FloodCell_qy;
					h_FloodCells->timeStep[*h_xmachine_memory_FloodCell_count] = FloodCell_timeStep;
					h_FloodCells->minh_loc[*h_xmachine_memory_FloodCell_count] = FloodCell_minh_loc;
					h_FloodCells->hFace_E[*h_xmachine_memory_FloodCell_count] = FloodCell_hFace_E;
					h_FloodCells->etFace_E[*h_xmachine_memory_FloodCell_count] = FloodCell_etFace_E;
					h_FloodCells->qxFace_E[*h_xmachine_memory_FloodCell_count] = FloodCell_qxFace_E;
					h_FloodCells->qyFace_E[*h_xmachine_memory_FloodCell_count] = FloodCell_qyFace_E;
					h_FloodCells->hFace_W[*h_xmachine_memory_FloodCell_count] = FloodCell_hFace_W;
					h_FloodCells->etFace_W[*h_xmachine_memory_FloodCell_count] = FloodCell_etFace_W;
					h_FloodCells->qxFace_W[*h_xmachine_memory_FloodCell_count] = FloodCell_qxFace_W;
					h_FloodCells->qyFace_W[*h_xmachine_memory_FloodCell_count] = FloodCell_qyFace_W;
					h_FloodCells->hFace_N[*h_xmachine_memory_FloodCell_count] = FloodCell_hFace_N;
					h_FloodCells->etFace_N[*h_xmachine_memory_FloodCell_count] = FloodCell_etFace_N;
					h_FloodCells->qxFace_N[*h_xmachine_memory_FloodCell_count] = FloodCell_qxFace_N;
					h_FloodCells->qyFace_N[*h_xmachine_memory_FloodCell_count] = FloodCell_qyFace_N;
					h_FloodCells->hFace_S[*h_xmachine_memory_FloodCell_count] = FloodCell_hFace_S;
					h_FloodCells->etFace_S[*h_xmachine_memory_FloodCell_count] = FloodCell_etFace_S;
					h_FloodCells->qxFace_S[*h_xmachine_memory_FloodCell_count] = FloodCell_qxFace_S;
					h_FloodCells->qyFace_S[*h_xmachine_memory_FloodCell_count] = FloodCell_qyFace_S;
					h_FloodCells->nm_rough[*h_xmachine_memory_FloodCell_count] = FloodCell_nm_rough;
					(*h_xmachine_memory_FloodCell_count) ++;	
				}
				else if(strcmp(agentname, "agent") == 0)
				{
					if (*h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent agent exceeded whilst reading data\n", xmachine_memory_agent_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_agents->x[*h_xmachine_memory_agent_count] = agent_x;//Check maximum x value
                    if(agent_maximum.x < agent_x)
                        agent_maximum.x = (float)agent_x;
                    //Check minimum x value
                    if(agent_minimum.x > agent_x)
                        agent_minimum.x = (float)agent_x;
                    
					h_agents->y[*h_xmachine_memory_agent_count] = agent_y;//Check maximum y value
                    if(agent_maximum.y < agent_y)
                        agent_maximum.y = (float)agent_y;
                    //Check minimum y value
                    if(agent_minimum.y > agent_y)
                        agent_minimum.y = (float)agent_y;
                    
					h_agents->velx[*h_xmachine_memory_agent_count] = agent_velx;
					h_agents->vely[*h_xmachine_memory_agent_count] = agent_vely;
					h_agents->steer_x[*h_xmachine_memory_agent_count] = agent_steer_x;
					h_agents->steer_y[*h_xmachine_memory_agent_count] = agent_steer_y;
					h_agents->height[*h_xmachine_memory_agent_count] = agent_height;
					h_agents->exit_no[*h_xmachine_memory_agent_count] = agent_exit_no;
					h_agents->speed[*h_xmachine_memory_agent_count] = agent_speed;
					h_agents->lod[*h_xmachine_memory_agent_count] = agent_lod;
					h_agents->animate[*h_xmachine_memory_agent_count] = agent_animate;
					h_agents->animate_dir[*h_xmachine_memory_agent_count] = agent_animate_dir;
					h_agents->HR_state[*h_xmachine_memory_agent_count] = agent_HR_state;
					h_agents->hero_status[*h_xmachine_memory_agent_count] = agent_hero_status;
					h_agents->pickup_time[*h_xmachine_memory_agent_count] = agent_pickup_time;
					h_agents->drop_time[*h_xmachine_memory_agent_count] = agent_drop_time;
					h_agents->carry_sandbag[*h_xmachine_memory_agent_count] = agent_carry_sandbag;
					h_agents->HR[*h_xmachine_memory_agent_count] = agent_HR;
					h_agents->dt_ped[*h_xmachine_memory_agent_count] = agent_dt_ped;
					h_agents->d_water[*h_xmachine_memory_agent_count] = agent_d_water;
					h_agents->v_water[*h_xmachine_memory_agent_count] = agent_v_water;
					h_agents->body_height[*h_xmachine_memory_agent_count] = agent_body_height;
					h_agents->body_mass[*h_xmachine_memory_agent_count] = agent_body_mass;
					h_agents->gender[*h_xmachine_memory_agent_count] = agent_gender;
					h_agents->stability_state[*h_xmachine_memory_agent_count] = agent_stability_state;
					h_agents->motion_speed[*h_xmachine_memory_agent_count] = agent_motion_speed;
					h_agents->age[*h_xmachine_memory_agent_count] = agent_age;
					h_agents->excitement_speed[*h_xmachine_memory_agent_count] = agent_excitement_speed;
					h_agents->dir_times[*h_xmachine_memory_agent_count] = agent_dir_times;
					h_agents->rejected_exit1[*h_xmachine_memory_agent_count] = agent_rejected_exit1;
					h_agents->rejected_exit2[*h_xmachine_memory_agent_count] = agent_rejected_exit2;
					h_agents->rejected_exit3[*h_xmachine_memory_agent_count] = agent_rejected_exit3;
					h_agents->rejected_exit4[*h_xmachine_memory_agent_count] = agent_rejected_exit4;
					h_agents->rejected_exit5[*h_xmachine_memory_agent_count] = agent_rejected_exit5;
					(*h_xmachine_memory_agent_count) ++;	
				}
				else if(strcmp(agentname, "navmap") == 0)
				{
					if (*h_xmachine_memory_navmap_count > xmachine_memory_navmap_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent navmap exceeded whilst reading data\n", xmachine_memory_navmap_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_navmaps->x[*h_xmachine_memory_navmap_count] = navmap_x;//Check maximum x value
                    if(agent_maximum.x < navmap_x)
                        agent_maximum.x = (float)navmap_x;
                    //Check minimum x value
                    if(agent_minimum.x > navmap_x)
                        agent_minimum.x = (float)navmap_x;
                    
					h_navmaps->y[*h_xmachine_memory_navmap_count] = navmap_y;//Check maximum y value
                    if(agent_maximum.y < navmap_y)
                        agent_maximum.y = (float)navmap_y;
                    //Check minimum y value
                    if(agent_minimum.y > navmap_y)
                        agent_minimum.y = (float)navmap_y;
                    
					h_navmaps->z0[*h_xmachine_memory_navmap_count] = navmap_z0;
					h_navmaps->h[*h_xmachine_memory_navmap_count] = navmap_h;
					h_navmaps->qx[*h_xmachine_memory_navmap_count] = navmap_qx;
					h_navmaps->qy[*h_xmachine_memory_navmap_count] = navmap_qy;
					h_navmaps->exit_no[*h_xmachine_memory_navmap_count] = navmap_exit_no;
					h_navmaps->height[*h_xmachine_memory_navmap_count] = navmap_height;
					h_navmaps->collision_x[*h_xmachine_memory_navmap_count] = navmap_collision_x;
					h_navmaps->collision_y[*h_xmachine_memory_navmap_count] = navmap_collision_y;
					h_navmaps->exit0_x[*h_xmachine_memory_navmap_count] = navmap_exit0_x;
					h_navmaps->exit0_y[*h_xmachine_memory_navmap_count] = navmap_exit0_y;
					h_navmaps->exit1_x[*h_xmachine_memory_navmap_count] = navmap_exit1_x;
					h_navmaps->exit1_y[*h_xmachine_memory_navmap_count] = navmap_exit1_y;
					h_navmaps->exit2_x[*h_xmachine_memory_navmap_count] = navmap_exit2_x;
					h_navmaps->exit2_y[*h_xmachine_memory_navmap_count] = navmap_exit2_y;
					h_navmaps->exit3_x[*h_xmachine_memory_navmap_count] = navmap_exit3_x;
					h_navmaps->exit3_y[*h_xmachine_memory_navmap_count] = navmap_exit3_y;
					h_navmaps->exit4_x[*h_xmachine_memory_navmap_count] = navmap_exit4_x;
					h_navmaps->exit4_y[*h_xmachine_memory_navmap_count] = navmap_exit4_y;
					h_navmaps->exit5_x[*h_xmachine_memory_navmap_count] = navmap_exit5_x;
					h_navmaps->exit5_y[*h_xmachine_memory_navmap_count] = navmap_exit5_y;
					h_navmaps->exit6_x[*h_xmachine_memory_navmap_count] = navmap_exit6_x;
					h_navmaps->exit6_y[*h_xmachine_memory_navmap_count] = navmap_exit6_y;
					h_navmaps->exit7_x[*h_xmachine_memory_navmap_count] = navmap_exit7_x;
					h_navmaps->exit7_y[*h_xmachine_memory_navmap_count] = navmap_exit7_y;
					h_navmaps->exit8_x[*h_xmachine_memory_navmap_count] = navmap_exit8_x;
					h_navmaps->exit8_y[*h_xmachine_memory_navmap_count] = navmap_exit8_y;
					h_navmaps->exit9_x[*h_xmachine_memory_navmap_count] = navmap_exit9_x;
					h_navmaps->exit9_y[*h_xmachine_memory_navmap_count] = navmap_exit9_y;
					h_navmaps->drop_point[*h_xmachine_memory_navmap_count] = navmap_drop_point;
					h_navmaps->sandbag_capacity[*h_xmachine_memory_navmap_count] = navmap_sandbag_capacity;
					h_navmaps->nm_rough[*h_xmachine_memory_navmap_count] = navmap_nm_rough;
					h_navmaps->evac_counter[*h_xmachine_memory_navmap_count] = navmap_evac_counter;
					(*h_xmachine_memory_navmap_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                FloodCell_inDomain = 0;
                FloodCell_x = 0;
                FloodCell_y = 0;
                FloodCell_z0 = 0;
                FloodCell_h = 0;
                FloodCell_qx = 0;
                FloodCell_qy = 0;
                FloodCell_timeStep = 0;
                FloodCell_minh_loc = 0;
                FloodCell_hFace_E = 0;
                FloodCell_etFace_E = 0;
                FloodCell_qxFace_E = 0;
                FloodCell_qyFace_E = 0;
                FloodCell_hFace_W = 0;
                FloodCell_etFace_W = 0;
                FloodCell_qxFace_W = 0;
                FloodCell_qyFace_W = 0;
                FloodCell_hFace_N = 0;
                FloodCell_etFace_N = 0;
                FloodCell_qxFace_N = 0;
                FloodCell_qyFace_N = 0;
                FloodCell_hFace_S = 0;
                FloodCell_etFace_S = 0;
                FloodCell_qxFace_S = 0;
                FloodCell_qyFace_S = 0;
                FloodCell_nm_rough = 0.01100f;
                agent_x = 0;
                agent_y = 0;
                agent_velx = 0;
                agent_vely = 0;
                agent_steer_x = 0;
                agent_steer_y = 0;
                agent_height = 0;
                agent_exit_no = 0;
                agent_speed = 0;
                agent_lod = 0;
                agent_animate = 0;
                agent_animate_dir = 0;
                agent_HR_state = 0;
                agent_hero_status = 0;
                agent_pickup_time = 0;
                agent_drop_time = 0;
                agent_carry_sandbag = 0;
                agent_HR = 0;
                agent_dt_ped = 0;
                agent_d_water = 0;
                agent_v_water = 0;
                agent_body_height = 0;
                agent_body_mass = 0;
                agent_gender = 0;
                agent_stability_state = 0;
                agent_motion_speed = 0;
                agent_age = 0;
                agent_excitement_speed = 0;
                agent_dir_times = 0;
                agent_rejected_exit1 = 0;
                agent_rejected_exit2 = 0;
                agent_rejected_exit3 = 0;
                agent_rejected_exit4 = 0;
                agent_rejected_exit5 = 0;
                navmap_x = 0;
                navmap_y = 0;
                navmap_z0 = 0;
                navmap_h = 0;
                navmap_qx = 0;
                navmap_qy = 0;
                navmap_exit_no = 0;
                navmap_height = 0;
                navmap_collision_x = 0;
                navmap_collision_y = 0;
                navmap_exit0_x = 0;
                navmap_exit0_y = 0;
                navmap_exit1_x = 0;
                navmap_exit1_y = 0;
                navmap_exit2_x = 0;
                navmap_exit2_y = 0;
                navmap_exit3_x = 0;
                navmap_exit3_y = 0;
                navmap_exit4_x = 0;
                navmap_exit4_y = 0;
                navmap_exit5_x = 0;
                navmap_exit5_y = 0;
                navmap_exit6_x = 0;
                navmap_exit6_y = 0;
                navmap_exit7_x = 0;
                navmap_exit7_y = 0;
                navmap_exit8_x = 0;
                navmap_exit8_y = 0;
                navmap_exit9_x = 0;
                navmap_exit9_y = 0;
                navmap_drop_point = 0;
                navmap_sandbag_capacity = 0;
                navmap_nm_rough = 0;
                navmap_evac_counter = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "inDomain") == 0) in_FloodCell_inDomain = 1;
			if(strcmp(buffer, "/inDomain") == 0) in_FloodCell_inDomain = 0;
			if(strcmp(buffer, "x") == 0) in_FloodCell_x = 1;
			if(strcmp(buffer, "/x") == 0) in_FloodCell_x = 0;
			if(strcmp(buffer, "y") == 0) in_FloodCell_y = 1;
			if(strcmp(buffer, "/y") == 0) in_FloodCell_y = 0;
			if(strcmp(buffer, "z0") == 0) in_FloodCell_z0 = 1;
			if(strcmp(buffer, "/z0") == 0) in_FloodCell_z0 = 0;
			if(strcmp(buffer, "h") == 0) in_FloodCell_h = 1;
			if(strcmp(buffer, "/h") == 0) in_FloodCell_h = 0;
			if(strcmp(buffer, "qx") == 0) in_FloodCell_qx = 1;
			if(strcmp(buffer, "/qx") == 0) in_FloodCell_qx = 0;
			if(strcmp(buffer, "qy") == 0) in_FloodCell_qy = 1;
			if(strcmp(buffer, "/qy") == 0) in_FloodCell_qy = 0;
			if(strcmp(buffer, "timeStep") == 0) in_FloodCell_timeStep = 1;
			if(strcmp(buffer, "/timeStep") == 0) in_FloodCell_timeStep = 0;
			if(strcmp(buffer, "minh_loc") == 0) in_FloodCell_minh_loc = 1;
			if(strcmp(buffer, "/minh_loc") == 0) in_FloodCell_minh_loc = 0;
			if(strcmp(buffer, "hFace_E") == 0) in_FloodCell_hFace_E = 1;
			if(strcmp(buffer, "/hFace_E") == 0) in_FloodCell_hFace_E = 0;
			if(strcmp(buffer, "etFace_E") == 0) in_FloodCell_etFace_E = 1;
			if(strcmp(buffer, "/etFace_E") == 0) in_FloodCell_etFace_E = 0;
			if(strcmp(buffer, "qxFace_E") == 0) in_FloodCell_qxFace_E = 1;
			if(strcmp(buffer, "/qxFace_E") == 0) in_FloodCell_qxFace_E = 0;
			if(strcmp(buffer, "qyFace_E") == 0) in_FloodCell_qyFace_E = 1;
			if(strcmp(buffer, "/qyFace_E") == 0) in_FloodCell_qyFace_E = 0;
			if(strcmp(buffer, "hFace_W") == 0) in_FloodCell_hFace_W = 1;
			if(strcmp(buffer, "/hFace_W") == 0) in_FloodCell_hFace_W = 0;
			if(strcmp(buffer, "etFace_W") == 0) in_FloodCell_etFace_W = 1;
			if(strcmp(buffer, "/etFace_W") == 0) in_FloodCell_etFace_W = 0;
			if(strcmp(buffer, "qxFace_W") == 0) in_FloodCell_qxFace_W = 1;
			if(strcmp(buffer, "/qxFace_W") == 0) in_FloodCell_qxFace_W = 0;
			if(strcmp(buffer, "qyFace_W") == 0) in_FloodCell_qyFace_W = 1;
			if(strcmp(buffer, "/qyFace_W") == 0) in_FloodCell_qyFace_W = 0;
			if(strcmp(buffer, "hFace_N") == 0) in_FloodCell_hFace_N = 1;
			if(strcmp(buffer, "/hFace_N") == 0) in_FloodCell_hFace_N = 0;
			if(strcmp(buffer, "etFace_N") == 0) in_FloodCell_etFace_N = 1;
			if(strcmp(buffer, "/etFace_N") == 0) in_FloodCell_etFace_N = 0;
			if(strcmp(buffer, "qxFace_N") == 0) in_FloodCell_qxFace_N = 1;
			if(strcmp(buffer, "/qxFace_N") == 0) in_FloodCell_qxFace_N = 0;
			if(strcmp(buffer, "qyFace_N") == 0) in_FloodCell_qyFace_N = 1;
			if(strcmp(buffer, "/qyFace_N") == 0) in_FloodCell_qyFace_N = 0;
			if(strcmp(buffer, "hFace_S") == 0) in_FloodCell_hFace_S = 1;
			if(strcmp(buffer, "/hFace_S") == 0) in_FloodCell_hFace_S = 0;
			if(strcmp(buffer, "etFace_S") == 0) in_FloodCell_etFace_S = 1;
			if(strcmp(buffer, "/etFace_S") == 0) in_FloodCell_etFace_S = 0;
			if(strcmp(buffer, "qxFace_S") == 0) in_FloodCell_qxFace_S = 1;
			if(strcmp(buffer, "/qxFace_S") == 0) in_FloodCell_qxFace_S = 0;
			if(strcmp(buffer, "qyFace_S") == 0) in_FloodCell_qyFace_S = 1;
			if(strcmp(buffer, "/qyFace_S") == 0) in_FloodCell_qyFace_S = 0;
			if(strcmp(buffer, "nm_rough") == 0) in_FloodCell_nm_rough = 1;
			if(strcmp(buffer, "/nm_rough") == 0) in_FloodCell_nm_rough = 0;
			if(strcmp(buffer, "x") == 0) in_agent_x = 1;
			if(strcmp(buffer, "/x") == 0) in_agent_x = 0;
			if(strcmp(buffer, "y") == 0) in_agent_y = 1;
			if(strcmp(buffer, "/y") == 0) in_agent_y = 0;
			if(strcmp(buffer, "velx") == 0) in_agent_velx = 1;
			if(strcmp(buffer, "/velx") == 0) in_agent_velx = 0;
			if(strcmp(buffer, "vely") == 0) in_agent_vely = 1;
			if(strcmp(buffer, "/vely") == 0) in_agent_vely = 0;
			if(strcmp(buffer, "steer_x") == 0) in_agent_steer_x = 1;
			if(strcmp(buffer, "/steer_x") == 0) in_agent_steer_x = 0;
			if(strcmp(buffer, "steer_y") == 0) in_agent_steer_y = 1;
			if(strcmp(buffer, "/steer_y") == 0) in_agent_steer_y = 0;
			if(strcmp(buffer, "height") == 0) in_agent_height = 1;
			if(strcmp(buffer, "/height") == 0) in_agent_height = 0;
			if(strcmp(buffer, "exit_no") == 0) in_agent_exit_no = 1;
			if(strcmp(buffer, "/exit_no") == 0) in_agent_exit_no = 0;
			if(strcmp(buffer, "speed") == 0) in_agent_speed = 1;
			if(strcmp(buffer, "/speed") == 0) in_agent_speed = 0;
			if(strcmp(buffer, "lod") == 0) in_agent_lod = 1;
			if(strcmp(buffer, "/lod") == 0) in_agent_lod = 0;
			if(strcmp(buffer, "animate") == 0) in_agent_animate = 1;
			if(strcmp(buffer, "/animate") == 0) in_agent_animate = 0;
			if(strcmp(buffer, "animate_dir") == 0) in_agent_animate_dir = 1;
			if(strcmp(buffer, "/animate_dir") == 0) in_agent_animate_dir = 0;
			if(strcmp(buffer, "HR_state") == 0) in_agent_HR_state = 1;
			if(strcmp(buffer, "/HR_state") == 0) in_agent_HR_state = 0;
			if(strcmp(buffer, "hero_status") == 0) in_agent_hero_status = 1;
			if(strcmp(buffer, "/hero_status") == 0) in_agent_hero_status = 0;
			if(strcmp(buffer, "pickup_time") == 0) in_agent_pickup_time = 1;
			if(strcmp(buffer, "/pickup_time") == 0) in_agent_pickup_time = 0;
			if(strcmp(buffer, "drop_time") == 0) in_agent_drop_time = 1;
			if(strcmp(buffer, "/drop_time") == 0) in_agent_drop_time = 0;
			if(strcmp(buffer, "carry_sandbag") == 0) in_agent_carry_sandbag = 1;
			if(strcmp(buffer, "/carry_sandbag") == 0) in_agent_carry_sandbag = 0;
			if(strcmp(buffer, "HR") == 0) in_agent_HR = 1;
			if(strcmp(buffer, "/HR") == 0) in_agent_HR = 0;
			if(strcmp(buffer, "dt_ped") == 0) in_agent_dt_ped = 1;
			if(strcmp(buffer, "/dt_ped") == 0) in_agent_dt_ped = 0;
			if(strcmp(buffer, "d_water") == 0) in_agent_d_water = 1;
			if(strcmp(buffer, "/d_water") == 0) in_agent_d_water = 0;
			if(strcmp(buffer, "v_water") == 0) in_agent_v_water = 1;
			if(strcmp(buffer, "/v_water") == 0) in_agent_v_water = 0;
			if(strcmp(buffer, "body_height") == 0) in_agent_body_height = 1;
			if(strcmp(buffer, "/body_height") == 0) in_agent_body_height = 0;
			if(strcmp(buffer, "body_mass") == 0) in_agent_body_mass = 1;
			if(strcmp(buffer, "/body_mass") == 0) in_agent_body_mass = 0;
			if(strcmp(buffer, "gender") == 0) in_agent_gender = 1;
			if(strcmp(buffer, "/gender") == 0) in_agent_gender = 0;
			if(strcmp(buffer, "stability_state") == 0) in_agent_stability_state = 1;
			if(strcmp(buffer, "/stability_state") == 0) in_agent_stability_state = 0;
			if(strcmp(buffer, "motion_speed") == 0) in_agent_motion_speed = 1;
			if(strcmp(buffer, "/motion_speed") == 0) in_agent_motion_speed = 0;
			if(strcmp(buffer, "age") == 0) in_agent_age = 1;
			if(strcmp(buffer, "/age") == 0) in_agent_age = 0;
			if(strcmp(buffer, "excitement_speed") == 0) in_agent_excitement_speed = 1;
			if(strcmp(buffer, "/excitement_speed") == 0) in_agent_excitement_speed = 0;
			if(strcmp(buffer, "dir_times") == 0) in_agent_dir_times = 1;
			if(strcmp(buffer, "/dir_times") == 0) in_agent_dir_times = 0;
			if(strcmp(buffer, "rejected_exit1") == 0) in_agent_rejected_exit1 = 1;
			if(strcmp(buffer, "/rejected_exit1") == 0) in_agent_rejected_exit1 = 0;
			if(strcmp(buffer, "rejected_exit2") == 0) in_agent_rejected_exit2 = 1;
			if(strcmp(buffer, "/rejected_exit2") == 0) in_agent_rejected_exit2 = 0;
			if(strcmp(buffer, "rejected_exit3") == 0) in_agent_rejected_exit3 = 1;
			if(strcmp(buffer, "/rejected_exit3") == 0) in_agent_rejected_exit3 = 0;
			if(strcmp(buffer, "rejected_exit4") == 0) in_agent_rejected_exit4 = 1;
			if(strcmp(buffer, "/rejected_exit4") == 0) in_agent_rejected_exit4 = 0;
			if(strcmp(buffer, "rejected_exit5") == 0) in_agent_rejected_exit5 = 1;
			if(strcmp(buffer, "/rejected_exit5") == 0) in_agent_rejected_exit5 = 0;
			if(strcmp(buffer, "x") == 0) in_navmap_x = 1;
			if(strcmp(buffer, "/x") == 0) in_navmap_x = 0;
			if(strcmp(buffer, "y") == 0) in_navmap_y = 1;
			if(strcmp(buffer, "/y") == 0) in_navmap_y = 0;
			if(strcmp(buffer, "z0") == 0) in_navmap_z0 = 1;
			if(strcmp(buffer, "/z0") == 0) in_navmap_z0 = 0;
			if(strcmp(buffer, "h") == 0) in_navmap_h = 1;
			if(strcmp(buffer, "/h") == 0) in_navmap_h = 0;
			if(strcmp(buffer, "qx") == 0) in_navmap_qx = 1;
			if(strcmp(buffer, "/qx") == 0) in_navmap_qx = 0;
			if(strcmp(buffer, "qy") == 0) in_navmap_qy = 1;
			if(strcmp(buffer, "/qy") == 0) in_navmap_qy = 0;
			if(strcmp(buffer, "exit_no") == 0) in_navmap_exit_no = 1;
			if(strcmp(buffer, "/exit_no") == 0) in_navmap_exit_no = 0;
			if(strcmp(buffer, "height") == 0) in_navmap_height = 1;
			if(strcmp(buffer, "/height") == 0) in_navmap_height = 0;
			if(strcmp(buffer, "collision_x") == 0) in_navmap_collision_x = 1;
			if(strcmp(buffer, "/collision_x") == 0) in_navmap_collision_x = 0;
			if(strcmp(buffer, "collision_y") == 0) in_navmap_collision_y = 1;
			if(strcmp(buffer, "/collision_y") == 0) in_navmap_collision_y = 0;
			if(strcmp(buffer, "exit0_x") == 0) in_navmap_exit0_x = 1;
			if(strcmp(buffer, "/exit0_x") == 0) in_navmap_exit0_x = 0;
			if(strcmp(buffer, "exit0_y") == 0) in_navmap_exit0_y = 1;
			if(strcmp(buffer, "/exit0_y") == 0) in_navmap_exit0_y = 0;
			if(strcmp(buffer, "exit1_x") == 0) in_navmap_exit1_x = 1;
			if(strcmp(buffer, "/exit1_x") == 0) in_navmap_exit1_x = 0;
			if(strcmp(buffer, "exit1_y") == 0) in_navmap_exit1_y = 1;
			if(strcmp(buffer, "/exit1_y") == 0) in_navmap_exit1_y = 0;
			if(strcmp(buffer, "exit2_x") == 0) in_navmap_exit2_x = 1;
			if(strcmp(buffer, "/exit2_x") == 0) in_navmap_exit2_x = 0;
			if(strcmp(buffer, "exit2_y") == 0) in_navmap_exit2_y = 1;
			if(strcmp(buffer, "/exit2_y") == 0) in_navmap_exit2_y = 0;
			if(strcmp(buffer, "exit3_x") == 0) in_navmap_exit3_x = 1;
			if(strcmp(buffer, "/exit3_x") == 0) in_navmap_exit3_x = 0;
			if(strcmp(buffer, "exit3_y") == 0) in_navmap_exit3_y = 1;
			if(strcmp(buffer, "/exit3_y") == 0) in_navmap_exit3_y = 0;
			if(strcmp(buffer, "exit4_x") == 0) in_navmap_exit4_x = 1;
			if(strcmp(buffer, "/exit4_x") == 0) in_navmap_exit4_x = 0;
			if(strcmp(buffer, "exit4_y") == 0) in_navmap_exit4_y = 1;
			if(strcmp(buffer, "/exit4_y") == 0) in_navmap_exit4_y = 0;
			if(strcmp(buffer, "exit5_x") == 0) in_navmap_exit5_x = 1;
			if(strcmp(buffer, "/exit5_x") == 0) in_navmap_exit5_x = 0;
			if(strcmp(buffer, "exit5_y") == 0) in_navmap_exit5_y = 1;
			if(strcmp(buffer, "/exit5_y") == 0) in_navmap_exit5_y = 0;
			if(strcmp(buffer, "exit6_x") == 0) in_navmap_exit6_x = 1;
			if(strcmp(buffer, "/exit6_x") == 0) in_navmap_exit6_x = 0;
			if(strcmp(buffer, "exit6_y") == 0) in_navmap_exit6_y = 1;
			if(strcmp(buffer, "/exit6_y") == 0) in_navmap_exit6_y = 0;
			if(strcmp(buffer, "exit7_x") == 0) in_navmap_exit7_x = 1;
			if(strcmp(buffer, "/exit7_x") == 0) in_navmap_exit7_x = 0;
			if(strcmp(buffer, "exit7_y") == 0) in_navmap_exit7_y = 1;
			if(strcmp(buffer, "/exit7_y") == 0) in_navmap_exit7_y = 0;
			if(strcmp(buffer, "exit8_x") == 0) in_navmap_exit8_x = 1;
			if(strcmp(buffer, "/exit8_x") == 0) in_navmap_exit8_x = 0;
			if(strcmp(buffer, "exit8_y") == 0) in_navmap_exit8_y = 1;
			if(strcmp(buffer, "/exit8_y") == 0) in_navmap_exit8_y = 0;
			if(strcmp(buffer, "exit9_x") == 0) in_navmap_exit9_x = 1;
			if(strcmp(buffer, "/exit9_x") == 0) in_navmap_exit9_x = 0;
			if(strcmp(buffer, "exit9_y") == 0) in_navmap_exit9_y = 1;
			if(strcmp(buffer, "/exit9_y") == 0) in_navmap_exit9_y = 0;
			if(strcmp(buffer, "drop_point") == 0) in_navmap_drop_point = 1;
			if(strcmp(buffer, "/drop_point") == 0) in_navmap_drop_point = 0;
			if(strcmp(buffer, "sandbag_capacity") == 0) in_navmap_sandbag_capacity = 1;
			if(strcmp(buffer, "/sandbag_capacity") == 0) in_navmap_sandbag_capacity = 0;
			if(strcmp(buffer, "nm_rough") == 0) in_navmap_nm_rough = 1;
			if(strcmp(buffer, "/nm_rough") == 0) in_navmap_nm_rough = 0;
			if(strcmp(buffer, "evac_counter") == 0) in_navmap_evac_counter = 1;
			if(strcmp(buffer, "/evac_counter") == 0) in_navmap_evac_counter = 0;
			
            /* environment variables */
            if(strcmp(buffer, "outputting_time") == 0) in_env_outputting_time = 1;
            if(strcmp(buffer, "/outputting_time") == 0) in_env_outputting_time = 0;
			if(strcmp(buffer, "outputting_time_interval") == 0) in_env_outputting_time_interval = 1;
            if(strcmp(buffer, "/outputting_time_interval") == 0) in_env_outputting_time_interval = 0;
			if(strcmp(buffer, "xmin") == 0) in_env_xmin = 1;
            if(strcmp(buffer, "/xmin") == 0) in_env_xmin = 0;
			if(strcmp(buffer, "xmax") == 0) in_env_xmax = 1;
            if(strcmp(buffer, "/xmax") == 0) in_env_xmax = 0;
			if(strcmp(buffer, "ymin") == 0) in_env_ymin = 1;
            if(strcmp(buffer, "/ymin") == 0) in_env_ymin = 0;
			if(strcmp(buffer, "ymax") == 0) in_env_ymax = 1;
            if(strcmp(buffer, "/ymax") == 0) in_env_ymax = 0;
			if(strcmp(buffer, "dt_ped") == 0) in_env_dt_ped = 1;
            if(strcmp(buffer, "/dt_ped") == 0) in_env_dt_ped = 0;
			if(strcmp(buffer, "dt_flood") == 0) in_env_dt_flood = 1;
            if(strcmp(buffer, "/dt_flood") == 0) in_env_dt_flood = 0;
			if(strcmp(buffer, "dt") == 0) in_env_dt = 1;
            if(strcmp(buffer, "/dt") == 0) in_env_dt = 0;
			if(strcmp(buffer, "auto_dt_on") == 0) in_env_auto_dt_on = 1;
            if(strcmp(buffer, "/auto_dt_on") == 0) in_env_auto_dt_on = 0;
			if(strcmp(buffer, "body_as_obstacle_on") == 0) in_env_body_as_obstacle_on = 1;
            if(strcmp(buffer, "/body_as_obstacle_on") == 0) in_env_body_as_obstacle_on = 0;
			if(strcmp(buffer, "ped_roughness_effect_on") == 0) in_env_ped_roughness_effect_on = 1;
            if(strcmp(buffer, "/ped_roughness_effect_on") == 0) in_env_ped_roughness_effect_on = 0;
			if(strcmp(buffer, "body_height") == 0) in_env_body_height = 1;
            if(strcmp(buffer, "/body_height") == 0) in_env_body_height = 0;
			if(strcmp(buffer, "init_speed") == 0) in_env_init_speed = 1;
            if(strcmp(buffer, "/init_speed") == 0) in_env_init_speed = 0;
			if(strcmp(buffer, "brisk_speed") == 0) in_env_brisk_speed = 1;
            if(strcmp(buffer, "/brisk_speed") == 0) in_env_brisk_speed = 0;
			if(strcmp(buffer, "sim_time") == 0) in_env_sim_time = 1;
            if(strcmp(buffer, "/sim_time") == 0) in_env_sim_time = 0;
			if(strcmp(buffer, "DXL") == 0) in_env_DXL = 1;
            if(strcmp(buffer, "/DXL") == 0) in_env_DXL = 0;
			if(strcmp(buffer, "DYL") == 0) in_env_DYL = 1;
            if(strcmp(buffer, "/DYL") == 0) in_env_DYL = 0;
			if(strcmp(buffer, "inflow_start_time") == 0) in_env_inflow_start_time = 1;
            if(strcmp(buffer, "/inflow_start_time") == 0) in_env_inflow_start_time = 0;
			if(strcmp(buffer, "inflow_peak_time") == 0) in_env_inflow_peak_time = 1;
            if(strcmp(buffer, "/inflow_peak_time") == 0) in_env_inflow_peak_time = 0;
			if(strcmp(buffer, "inflow_end_time") == 0) in_env_inflow_end_time = 1;
            if(strcmp(buffer, "/inflow_end_time") == 0) in_env_inflow_end_time = 0;
			if(strcmp(buffer, "inflow_initial_discharge") == 0) in_env_inflow_initial_discharge = 1;
            if(strcmp(buffer, "/inflow_initial_discharge") == 0) in_env_inflow_initial_discharge = 0;
			if(strcmp(buffer, "inflow_peak_discharge") == 0) in_env_inflow_peak_discharge = 1;
            if(strcmp(buffer, "/inflow_peak_discharge") == 0) in_env_inflow_peak_discharge = 0;
			if(strcmp(buffer, "inflow_end_discharge") == 0) in_env_inflow_end_discharge = 1;
            if(strcmp(buffer, "/inflow_end_discharge") == 0) in_env_inflow_end_discharge = 0;
			if(strcmp(buffer, "INFLOW_BOUNDARY") == 0) in_env_INFLOW_BOUNDARY = 1;
            if(strcmp(buffer, "/INFLOW_BOUNDARY") == 0) in_env_INFLOW_BOUNDARY = 0;
			if(strcmp(buffer, "BOUNDARY_EAST_STATUS") == 0) in_env_BOUNDARY_EAST_STATUS = 1;
            if(strcmp(buffer, "/BOUNDARY_EAST_STATUS") == 0) in_env_BOUNDARY_EAST_STATUS = 0;
			if(strcmp(buffer, "BOUNDARY_WEST_STATUS") == 0) in_env_BOUNDARY_WEST_STATUS = 1;
            if(strcmp(buffer, "/BOUNDARY_WEST_STATUS") == 0) in_env_BOUNDARY_WEST_STATUS = 0;
			if(strcmp(buffer, "BOUNDARY_NORTH_STATUS") == 0) in_env_BOUNDARY_NORTH_STATUS = 1;
            if(strcmp(buffer, "/BOUNDARY_NORTH_STATUS") == 0) in_env_BOUNDARY_NORTH_STATUS = 0;
			if(strcmp(buffer, "BOUNDARY_SOUTH_STATUS") == 0) in_env_BOUNDARY_SOUTH_STATUS = 1;
            if(strcmp(buffer, "/BOUNDARY_SOUTH_STATUS") == 0) in_env_BOUNDARY_SOUTH_STATUS = 0;
			if(strcmp(buffer, "x1_boundary") == 0) in_env_x1_boundary = 1;
            if(strcmp(buffer, "/x1_boundary") == 0) in_env_x1_boundary = 0;
			if(strcmp(buffer, "x2_boundary") == 0) in_env_x2_boundary = 1;
            if(strcmp(buffer, "/x2_boundary") == 0) in_env_x2_boundary = 0;
			if(strcmp(buffer, "y1_boundary") == 0) in_env_y1_boundary = 1;
            if(strcmp(buffer, "/y1_boundary") == 0) in_env_y1_boundary = 0;
			if(strcmp(buffer, "y2_boundary") == 0) in_env_y2_boundary = 1;
            if(strcmp(buffer, "/y2_boundary") == 0) in_env_y2_boundary = 0;
			if(strcmp(buffer, "init_depth_boundary") == 0) in_env_init_depth_boundary = 1;
            if(strcmp(buffer, "/init_depth_boundary") == 0) in_env_init_depth_boundary = 0;
			if(strcmp(buffer, "evacuation_on") == 0) in_env_evacuation_on = 1;
            if(strcmp(buffer, "/evacuation_on") == 0) in_env_evacuation_on = 0;
			if(strcmp(buffer, "walking_speed_reduction_in_water_on") == 0) in_env_walking_speed_reduction_in_water_on = 1;
            if(strcmp(buffer, "/walking_speed_reduction_in_water_on") == 0) in_env_walking_speed_reduction_in_water_on = 0;
			if(strcmp(buffer, "freeze_while_instable_on") == 0) in_env_freeze_while_instable_on = 1;
            if(strcmp(buffer, "/freeze_while_instable_on") == 0) in_env_freeze_while_instable_on = 0;
			if(strcmp(buffer, "evacuation_end_time") == 0) in_env_evacuation_end_time = 1;
            if(strcmp(buffer, "/evacuation_end_time") == 0) in_env_evacuation_end_time = 0;
			if(strcmp(buffer, "evacuation_start_time") == 0) in_env_evacuation_start_time = 1;
            if(strcmp(buffer, "/evacuation_start_time") == 0) in_env_evacuation_start_time = 0;
			if(strcmp(buffer, "emergency_exit_number") == 0) in_env_emergency_exit_number = 1;
            if(strcmp(buffer, "/emergency_exit_number") == 0) in_env_emergency_exit_number = 0;
			if(strcmp(buffer, "emer_alarm") == 0) in_env_emer_alarm = 1;
            if(strcmp(buffer, "/emer_alarm") == 0) in_env_emer_alarm = 0;
			if(strcmp(buffer, "HR") == 0) in_env_HR = 1;
            if(strcmp(buffer, "/HR") == 0) in_env_HR = 0;
			if(strcmp(buffer, "max_at_highest_risk") == 0) in_env_max_at_highest_risk = 1;
            if(strcmp(buffer, "/max_at_highest_risk") == 0) in_env_max_at_highest_risk = 0;
			if(strcmp(buffer, "max_at_low_risk") == 0) in_env_max_at_low_risk = 1;
            if(strcmp(buffer, "/max_at_low_risk") == 0) in_env_max_at_low_risk = 0;
			if(strcmp(buffer, "max_at_medium_risk") == 0) in_env_max_at_medium_risk = 1;
            if(strcmp(buffer, "/max_at_medium_risk") == 0) in_env_max_at_medium_risk = 0;
			if(strcmp(buffer, "max_at_high_risk") == 0) in_env_max_at_high_risk = 1;
            if(strcmp(buffer, "/max_at_high_risk") == 0) in_env_max_at_high_risk = 0;
			if(strcmp(buffer, "max_velocity") == 0) in_env_max_velocity = 1;
            if(strcmp(buffer, "/max_velocity") == 0) in_env_max_velocity = 0;
			if(strcmp(buffer, "max_depth") == 0) in_env_max_depth = 1;
            if(strcmp(buffer, "/max_depth") == 0) in_env_max_depth = 0;
			if(strcmp(buffer, "count_population") == 0) in_env_count_population = 1;
            if(strcmp(buffer, "/count_population") == 0) in_env_count_population = 0;
			if(strcmp(buffer, "count_heros") == 0) in_env_count_heros = 1;
            if(strcmp(buffer, "/count_heros") == 0) in_env_count_heros = 0;
			if(strcmp(buffer, "initial_population") == 0) in_env_initial_population = 1;
            if(strcmp(buffer, "/initial_population") == 0) in_env_initial_population = 0;
			if(strcmp(buffer, "evacuated_population") == 0) in_env_evacuated_population = 1;
            if(strcmp(buffer, "/evacuated_population") == 0) in_env_evacuated_population = 0;
			if(strcmp(buffer, "hero_percentage") == 0) in_env_hero_percentage = 1;
            if(strcmp(buffer, "/hero_percentage") == 0) in_env_hero_percentage = 0;
			if(strcmp(buffer, "hero_population") == 0) in_env_hero_population = 1;
            if(strcmp(buffer, "/hero_population") == 0) in_env_hero_population = 0;
			if(strcmp(buffer, "sandbagging_on") == 0) in_env_sandbagging_on = 1;
            if(strcmp(buffer, "/sandbagging_on") == 0) in_env_sandbagging_on = 0;
			if(strcmp(buffer, "sandbagging_start_time") == 0) in_env_sandbagging_start_time = 1;
            if(strcmp(buffer, "/sandbagging_start_time") == 0) in_env_sandbagging_start_time = 0;
			if(strcmp(buffer, "sandbagging_end_time") == 0) in_env_sandbagging_end_time = 1;
            if(strcmp(buffer, "/sandbagging_end_time") == 0) in_env_sandbagging_end_time = 0;
			if(strcmp(buffer, "sandbag_length") == 0) in_env_sandbag_length = 1;
            if(strcmp(buffer, "/sandbag_length") == 0) in_env_sandbag_length = 0;
			if(strcmp(buffer, "sandbag_height") == 0) in_env_sandbag_height = 1;
            if(strcmp(buffer, "/sandbag_height") == 0) in_env_sandbag_height = 0;
			if(strcmp(buffer, "sandbag_width") == 0) in_env_sandbag_width = 1;
            if(strcmp(buffer, "/sandbag_width") == 0) in_env_sandbag_width = 0;
			if(strcmp(buffer, "extended_length") == 0) in_env_extended_length = 1;
            if(strcmp(buffer, "/extended_length") == 0) in_env_extended_length = 0;
			if(strcmp(buffer, "sandbag_layers") == 0) in_env_sandbag_layers = 1;
            if(strcmp(buffer, "/sandbag_layers") == 0) in_env_sandbag_layers = 0;
			if(strcmp(buffer, "update_stopper") == 0) in_env_update_stopper = 1;
            if(strcmp(buffer, "/update_stopper") == 0) in_env_update_stopper = 0;
			if(strcmp(buffer, "dike_length") == 0) in_env_dike_length = 1;
            if(strcmp(buffer, "/dike_length") == 0) in_env_dike_length = 0;
			if(strcmp(buffer, "dike_height") == 0) in_env_dike_height = 1;
            if(strcmp(buffer, "/dike_height") == 0) in_env_dike_height = 0;
			if(strcmp(buffer, "dike_width") == 0) in_env_dike_width = 1;
            if(strcmp(buffer, "/dike_width") == 0) in_env_dike_width = 0;
			if(strcmp(buffer, "fill_cap") == 0) in_env_fill_cap = 1;
            if(strcmp(buffer, "/fill_cap") == 0) in_env_fill_cap = 0;
			if(strcmp(buffer, "pickup_point") == 0) in_env_pickup_point = 1;
            if(strcmp(buffer, "/pickup_point") == 0) in_env_pickup_point = 0;
			if(strcmp(buffer, "drop_point") == 0) in_env_drop_point = 1;
            if(strcmp(buffer, "/drop_point") == 0) in_env_drop_point = 0;
			if(strcmp(buffer, "pickup_duration") == 0) in_env_pickup_duration = 1;
            if(strcmp(buffer, "/pickup_duration") == 0) in_env_pickup_duration = 0;
			if(strcmp(buffer, "drop_duration") == 0) in_env_drop_duration = 1;
            if(strcmp(buffer, "/drop_duration") == 0) in_env_drop_duration = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT1") == 0) in_env_EMMISION_RATE_EXIT1 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT1") == 0) in_env_EMMISION_RATE_EXIT1 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT2") == 0) in_env_EMMISION_RATE_EXIT2 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT2") == 0) in_env_EMMISION_RATE_EXIT2 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT3") == 0) in_env_EMMISION_RATE_EXIT3 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT3") == 0) in_env_EMMISION_RATE_EXIT3 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT4") == 0) in_env_EMMISION_RATE_EXIT4 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT4") == 0) in_env_EMMISION_RATE_EXIT4 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT5") == 0) in_env_EMMISION_RATE_EXIT5 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT5") == 0) in_env_EMMISION_RATE_EXIT5 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT6") == 0) in_env_EMMISION_RATE_EXIT6 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT6") == 0) in_env_EMMISION_RATE_EXIT6 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT7") == 0) in_env_EMMISION_RATE_EXIT7 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT7") == 0) in_env_EMMISION_RATE_EXIT7 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT8") == 0) in_env_EMMISION_RATE_EXIT8 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT8") == 0) in_env_EMMISION_RATE_EXIT8 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT9") == 0) in_env_EMMISION_RATE_EXIT9 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT9") == 0) in_env_EMMISION_RATE_EXIT9 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT10") == 0) in_env_EMMISION_RATE_EXIT10 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT10") == 0) in_env_EMMISION_RATE_EXIT10 = 0;
			if(strcmp(buffer, "EXIT1_PROBABILITY") == 0) in_env_EXIT1_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT1_PROBABILITY") == 0) in_env_EXIT1_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT2_PROBABILITY") == 0) in_env_EXIT2_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT2_PROBABILITY") == 0) in_env_EXIT2_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT3_PROBABILITY") == 0) in_env_EXIT3_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT3_PROBABILITY") == 0) in_env_EXIT3_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT4_PROBABILITY") == 0) in_env_EXIT4_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT4_PROBABILITY") == 0) in_env_EXIT4_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT5_PROBABILITY") == 0) in_env_EXIT5_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT5_PROBABILITY") == 0) in_env_EXIT5_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT6_PROBABILITY") == 0) in_env_EXIT6_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT6_PROBABILITY") == 0) in_env_EXIT6_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT7_PROBABILITY") == 0) in_env_EXIT7_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT7_PROBABILITY") == 0) in_env_EXIT7_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT8_PROBABILITY") == 0) in_env_EXIT8_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT8_PROBABILITY") == 0) in_env_EXIT8_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT9_PROBABILITY") == 0) in_env_EXIT9_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT9_PROBABILITY") == 0) in_env_EXIT9_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT10_PROBABILITY") == 0) in_env_EXIT10_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT10_PROBABILITY") == 0) in_env_EXIT10_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT1_STATE") == 0) in_env_EXIT1_STATE = 1;
            if(strcmp(buffer, "/EXIT1_STATE") == 0) in_env_EXIT1_STATE = 0;
			if(strcmp(buffer, "EXIT2_STATE") == 0) in_env_EXIT2_STATE = 1;
            if(strcmp(buffer, "/EXIT2_STATE") == 0) in_env_EXIT2_STATE = 0;
			if(strcmp(buffer, "EXIT3_STATE") == 0) in_env_EXIT3_STATE = 1;
            if(strcmp(buffer, "/EXIT3_STATE") == 0) in_env_EXIT3_STATE = 0;
			if(strcmp(buffer, "EXIT4_STATE") == 0) in_env_EXIT4_STATE = 1;
            if(strcmp(buffer, "/EXIT4_STATE") == 0) in_env_EXIT4_STATE = 0;
			if(strcmp(buffer, "EXIT5_STATE") == 0) in_env_EXIT5_STATE = 1;
            if(strcmp(buffer, "/EXIT5_STATE") == 0) in_env_EXIT5_STATE = 0;
			if(strcmp(buffer, "EXIT6_STATE") == 0) in_env_EXIT6_STATE = 1;
            if(strcmp(buffer, "/EXIT6_STATE") == 0) in_env_EXIT6_STATE = 0;
			if(strcmp(buffer, "EXIT7_STATE") == 0) in_env_EXIT7_STATE = 1;
            if(strcmp(buffer, "/EXIT7_STATE") == 0) in_env_EXIT7_STATE = 0;
			if(strcmp(buffer, "EXIT8_STATE") == 0) in_env_EXIT8_STATE = 1;
            if(strcmp(buffer, "/EXIT8_STATE") == 0) in_env_EXIT8_STATE = 0;
			if(strcmp(buffer, "EXIT9_STATE") == 0) in_env_EXIT9_STATE = 1;
            if(strcmp(buffer, "/EXIT9_STATE") == 0) in_env_EXIT9_STATE = 0;
			if(strcmp(buffer, "EXIT10_STATE") == 0) in_env_EXIT10_STATE = 1;
            if(strcmp(buffer, "/EXIT10_STATE") == 0) in_env_EXIT10_STATE = 0;
			if(strcmp(buffer, "EXIT1_CELL_COUNT") == 0) in_env_EXIT1_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT1_CELL_COUNT") == 0) in_env_EXIT1_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT2_CELL_COUNT") == 0) in_env_EXIT2_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT2_CELL_COUNT") == 0) in_env_EXIT2_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT3_CELL_COUNT") == 0) in_env_EXIT3_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT3_CELL_COUNT") == 0) in_env_EXIT3_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT4_CELL_COUNT") == 0) in_env_EXIT4_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT4_CELL_COUNT") == 0) in_env_EXIT4_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT5_CELL_COUNT") == 0) in_env_EXIT5_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT5_CELL_COUNT") == 0) in_env_EXIT5_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT6_CELL_COUNT") == 0) in_env_EXIT6_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT6_CELL_COUNT") == 0) in_env_EXIT6_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT7_CELL_COUNT") == 0) in_env_EXIT7_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT7_CELL_COUNT") == 0) in_env_EXIT7_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT8_CELL_COUNT") == 0) in_env_EXIT8_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT8_CELL_COUNT") == 0) in_env_EXIT8_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT9_CELL_COUNT") == 0) in_env_EXIT9_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT9_CELL_COUNT") == 0) in_env_EXIT9_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT10_CELL_COUNT") == 0) in_env_EXIT10_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT10_CELL_COUNT") == 0) in_env_EXIT10_CELL_COUNT = 0;
			if(strcmp(buffer, "TIME_SCALER") == 0) in_env_TIME_SCALER = 1;
            if(strcmp(buffer, "/TIME_SCALER") == 0) in_env_TIME_SCALER = 0;
			if(strcmp(buffer, "STEER_WEIGHT") == 0) in_env_STEER_WEIGHT = 1;
            if(strcmp(buffer, "/STEER_WEIGHT") == 0) in_env_STEER_WEIGHT = 0;
			if(strcmp(buffer, "AVOID_WEIGHT") == 0) in_env_AVOID_WEIGHT = 1;
            if(strcmp(buffer, "/AVOID_WEIGHT") == 0) in_env_AVOID_WEIGHT = 0;
			if(strcmp(buffer, "COLLISION_WEIGHT") == 0) in_env_COLLISION_WEIGHT = 1;
            if(strcmp(buffer, "/COLLISION_WEIGHT") == 0) in_env_COLLISION_WEIGHT = 0;
			if(strcmp(buffer, "GOAL_WEIGHT") == 0) in_env_GOAL_WEIGHT = 1;
            if(strcmp(buffer, "/GOAL_WEIGHT") == 0) in_env_GOAL_WEIGHT = 0;
			if(strcmp(buffer, "PedHeight_60_110_probability") == 0) in_env_PedHeight_60_110_probability = 1;
            if(strcmp(buffer, "/PedHeight_60_110_probability") == 0) in_env_PedHeight_60_110_probability = 0;
			if(strcmp(buffer, "PedHeight_110_140_probability") == 0) in_env_PedHeight_110_140_probability = 1;
            if(strcmp(buffer, "/PedHeight_110_140_probability") == 0) in_env_PedHeight_110_140_probability = 0;
			if(strcmp(buffer, "PedHeight_140_163_probability") == 0) in_env_PedHeight_140_163_probability = 1;
            if(strcmp(buffer, "/PedHeight_140_163_probability") == 0) in_env_PedHeight_140_163_probability = 0;
			if(strcmp(buffer, "PedHeight_163_170_probability") == 0) in_env_PedHeight_163_170_probability = 1;
            if(strcmp(buffer, "/PedHeight_163_170_probability") == 0) in_env_PedHeight_163_170_probability = 0;
			if(strcmp(buffer, "PedHeight_170_186_probability") == 0) in_env_PedHeight_170_186_probability = 1;
            if(strcmp(buffer, "/PedHeight_170_186_probability") == 0) in_env_PedHeight_170_186_probability = 0;
			if(strcmp(buffer, "PedHeight_186_194_probability") == 0) in_env_PedHeight_186_194_probability = 1;
            if(strcmp(buffer, "/PedHeight_186_194_probability") == 0) in_env_PedHeight_186_194_probability = 0;
			if(strcmp(buffer, "PedHeight_194_210_probability") == 0) in_env_PedHeight_194_210_probability = 1;
            if(strcmp(buffer, "/PedHeight_194_210_probability") == 0) in_env_PedHeight_194_210_probability = 0;
			if(strcmp(buffer, "PedAge_10_17_probability") == 0) in_env_PedAge_10_17_probability = 1;
            if(strcmp(buffer, "/PedAge_10_17_probability") == 0) in_env_PedAge_10_17_probability = 0;
			if(strcmp(buffer, "PedAge_18_29_probability") == 0) in_env_PedAge_18_29_probability = 1;
            if(strcmp(buffer, "/PedAge_18_29_probability") == 0) in_env_PedAge_18_29_probability = 0;
			if(strcmp(buffer, "PedAge_30_39_probability") == 0) in_env_PedAge_30_39_probability = 1;
            if(strcmp(buffer, "/PedAge_30_39_probability") == 0) in_env_PedAge_30_39_probability = 0;
			if(strcmp(buffer, "PedAge_40_49_probability") == 0) in_env_PedAge_40_49_probability = 1;
            if(strcmp(buffer, "/PedAge_40_49_probability") == 0) in_env_PedAge_40_49_probability = 0;
			if(strcmp(buffer, "PedAge_50_59_probability") == 0) in_env_PedAge_50_59_probability = 1;
            if(strcmp(buffer, "/PedAge_50_59_probability") == 0) in_env_PedAge_50_59_probability = 0;
			if(strcmp(buffer, "PedAge_60_69_probability") == 0) in_env_PedAge_60_69_probability = 1;
            if(strcmp(buffer, "/PedAge_60_69_probability") == 0) in_env_PedAge_60_69_probability = 0;
			if(strcmp(buffer, "PedAge_70_79_probability") == 0) in_env_PedAge_70_79_probability = 1;
            if(strcmp(buffer, "/PedAge_70_79_probability") == 0) in_env_PedAge_70_79_probability = 0;
			if(strcmp(buffer, "excluded_age_probability") == 0) in_env_excluded_age_probability = 1;
            if(strcmp(buffer, "/excluded_age_probability") == 0) in_env_excluded_age_probability = 0;
			if(strcmp(buffer, "gender_female_probability") == 0) in_env_gender_female_probability = 1;
            if(strcmp(buffer, "/gender_female_probability") == 0) in_env_gender_female_probability = 0;
			if(strcmp(buffer, "gender_male_probability") == 0) in_env_gender_male_probability = 1;
            if(strcmp(buffer, "/gender_male_probability") == 0) in_env_gender_male_probability = 0;
			if(strcmp(buffer, "SCALE_FACTOR") == 0) in_env_SCALE_FACTOR = 1;
            if(strcmp(buffer, "/SCALE_FACTOR") == 0) in_env_SCALE_FACTOR = 0;
			if(strcmp(buffer, "I_SCALER") == 0) in_env_I_SCALER = 1;
            if(strcmp(buffer, "/I_SCALER") == 0) in_env_I_SCALER = 0;
			if(strcmp(buffer, "MIN_DISTANCE") == 0) in_env_MIN_DISTANCE = 1;
            if(strcmp(buffer, "/MIN_DISTANCE") == 0) in_env_MIN_DISTANCE = 0;
			if(strcmp(buffer, "excitement_on") == 0) in_env_excitement_on = 1;
            if(strcmp(buffer, "/excitement_on") == 0) in_env_excitement_on = 0;
			if(strcmp(buffer, "walk_run_switch") == 0) in_env_walk_run_switch = 1;
            if(strcmp(buffer, "/walk_run_switch") == 0) in_env_walk_run_switch = 0;
			if(strcmp(buffer, "preoccupying_on") == 0) in_env_preoccupying_on = 1;
            if(strcmp(buffer, "/preoccupying_on") == 0) in_env_preoccupying_on = 0;
			if(strcmp(buffer, "poly_hydrograph_on") == 0) in_env_poly_hydrograph_on = 1;
            if(strcmp(buffer, "/poly_hydrograph_on") == 0) in_env_poly_hydrograph_on = 0;
			if(strcmp(buffer, "stop_emission_on") == 0) in_env_stop_emission_on = 1;
            if(strcmp(buffer, "/stop_emission_on") == 0) in_env_stop_emission_on = 0;
			if(strcmp(buffer, "goto_emergency_exit_on") == 0) in_env_goto_emergency_exit_on = 1;
            if(strcmp(buffer, "/goto_emergency_exit_on") == 0) in_env_goto_emergency_exit_on = 0;
			if(strcmp(buffer, "escape_route_finder_on") == 0) in_env_escape_route_finder_on = 1;
            if(strcmp(buffer, "/escape_route_finder_on") == 0) in_env_escape_route_finder_on = 0;
			if(strcmp(buffer, "dir_times") == 0) in_env_dir_times = 1;
            if(strcmp(buffer, "/dir_times") == 0) in_env_dir_times = 0;
			if(strcmp(buffer, "no_return_on") == 0) in_env_no_return_on = 1;
            if(strcmp(buffer, "/no_return_on") == 0) in_env_no_return_on = 0;
			if(strcmp(buffer, "wdepth_perc_thresh") == 0) in_env_wdepth_perc_thresh = 1;
            if(strcmp(buffer, "/wdepth_perc_thresh") == 0) in_env_wdepth_perc_thresh = 0;
			if(strcmp(buffer, "follow_popular_exit_on") == 0) in_env_follow_popular_exit_on = 1;
            if(strcmp(buffer, "/follow_popular_exit_on") == 0) in_env_follow_popular_exit_on = 0;
			if(strcmp(buffer, "popular_exit") == 0) in_env_popular_exit = 1;
            if(strcmp(buffer, "/popular_exit") == 0) in_env_popular_exit = 0;
			

			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;

			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else if (in_xagent)
			{
				if(in_FloodCell_inDomain){
                    FloodCell_inDomain = (int) fpgu_strtol(buffer); 
                }
				if(in_FloodCell_x){
                    FloodCell_x = (int) fpgu_strtol(buffer); 
                }
				if(in_FloodCell_y){
                    FloodCell_y = (int) fpgu_strtol(buffer); 
                }
				if(in_FloodCell_z0){
                    FloodCell_z0 = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_h){
                    FloodCell_h = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qx){
                    FloodCell_qx = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qy){
                    FloodCell_qy = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_timeStep){
                    FloodCell_timeStep = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_minh_loc){
                    FloodCell_minh_loc = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_hFace_E){
                    FloodCell_hFace_E = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_etFace_E){
                    FloodCell_etFace_E = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qxFace_E){
                    FloodCell_qxFace_E = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qyFace_E){
                    FloodCell_qyFace_E = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_hFace_W){
                    FloodCell_hFace_W = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_etFace_W){
                    FloodCell_etFace_W = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qxFace_W){
                    FloodCell_qxFace_W = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qyFace_W){
                    FloodCell_qyFace_W = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_hFace_N){
                    FloodCell_hFace_N = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_etFace_N){
                    FloodCell_etFace_N = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qxFace_N){
                    FloodCell_qxFace_N = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qyFace_N){
                    FloodCell_qyFace_N = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_hFace_S){
                    FloodCell_hFace_S = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_etFace_S){
                    FloodCell_etFace_S = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qxFace_S){
                    FloodCell_qxFace_S = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qyFace_S){
                    FloodCell_qyFace_S = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_nm_rough){
                    FloodCell_nm_rough = (double) fpgu_strtod(buffer); 
                }
				if(in_agent_x){
                    agent_x = (float) fgpu_atof(buffer); 
                }
				if(in_agent_y){
                    agent_y = (float) fgpu_atof(buffer); 
                }
				if(in_agent_velx){
                    agent_velx = (float) fgpu_atof(buffer); 
                }
				if(in_agent_vely){
                    agent_vely = (float) fgpu_atof(buffer); 
                }
				if(in_agent_steer_x){
                    agent_steer_x = (float) fgpu_atof(buffer); 
                }
				if(in_agent_steer_y){
                    agent_steer_y = (float) fgpu_atof(buffer); 
                }
				if(in_agent_height){
                    agent_height = (float) fgpu_atof(buffer); 
                }
				if(in_agent_exit_no){
                    agent_exit_no = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_speed){
                    agent_speed = (float) fgpu_atof(buffer); 
                }
				if(in_agent_lod){
                    agent_lod = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_animate){
                    agent_animate = (float) fgpu_atof(buffer); 
                }
				if(in_agent_animate_dir){
                    agent_animate_dir = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_HR_state){
                    agent_HR_state = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_hero_status){
                    agent_hero_status = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_pickup_time){
                    agent_pickup_time = (double) fpgu_strtod(buffer); 
                }
				if(in_agent_drop_time){
                    agent_drop_time = (double) fpgu_strtod(buffer); 
                }
				if(in_agent_carry_sandbag){
                    agent_carry_sandbag = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_HR){
                    agent_HR = (double) fpgu_strtod(buffer); 
                }
				if(in_agent_dt_ped){
                    agent_dt_ped = (float) fgpu_atof(buffer); 
                }
				if(in_agent_d_water){
                    agent_d_water = (float) fgpu_atof(buffer); 
                }
				if(in_agent_v_water){
                    agent_v_water = (float) fgpu_atof(buffer); 
                }
				if(in_agent_body_height){
                    agent_body_height = (float) fgpu_atof(buffer); 
                }
				if(in_agent_body_mass){
                    agent_body_mass = (float) fgpu_atof(buffer); 
                }
				if(in_agent_gender){
                    agent_gender = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_stability_state){
                    agent_stability_state = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_motion_speed){
                    agent_motion_speed = (float) fgpu_atof(buffer); 
                }
				if(in_agent_age){
                    agent_age = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_excitement_speed){
                    agent_excitement_speed = (float) fgpu_atof(buffer); 
                }
				if(in_agent_dir_times){
                    agent_dir_times = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_rejected_exit1){
                    agent_rejected_exit1 = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_rejected_exit2){
                    agent_rejected_exit2 = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_rejected_exit3){
                    agent_rejected_exit3 = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_rejected_exit4){
                    agent_rejected_exit4 = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_rejected_exit5){
                    agent_rejected_exit5 = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_x){
                    navmap_x = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_y){
                    navmap_y = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_z0){
                    navmap_z0 = (double) fpgu_strtod(buffer); 
                }
				if(in_navmap_h){
                    navmap_h = (double) fpgu_strtod(buffer); 
                }
				if(in_navmap_qx){
                    navmap_qx = (double) fpgu_strtod(buffer); 
                }
				if(in_navmap_qy){
                    navmap_qy = (double) fpgu_strtod(buffer); 
                }
				if(in_navmap_exit_no){
                    navmap_exit_no = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_height){
                    navmap_height = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_collision_x){
                    navmap_collision_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_collision_y){
                    navmap_collision_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit0_x){
                    navmap_exit0_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit0_y){
                    navmap_exit0_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit1_x){
                    navmap_exit1_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit1_y){
                    navmap_exit1_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit2_x){
                    navmap_exit2_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit2_y){
                    navmap_exit2_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit3_x){
                    navmap_exit3_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit3_y){
                    navmap_exit3_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit4_x){
                    navmap_exit4_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit4_y){
                    navmap_exit4_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit5_x){
                    navmap_exit5_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit5_y){
                    navmap_exit5_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit6_x){
                    navmap_exit6_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit6_y){
                    navmap_exit6_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit7_x){
                    navmap_exit7_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit7_y){
                    navmap_exit7_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit8_x){
                    navmap_exit8_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit8_y){
                    navmap_exit8_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit9_x){
                    navmap_exit9_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit9_y){
                    navmap_exit9_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_drop_point){
                    navmap_drop_point = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_sandbag_capacity){
                    navmap_sandbag_capacity = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_nm_rough){
                    navmap_nm_rough = (double) fpgu_strtod(buffer); 
                }
				if(in_navmap_evac_counter){
                    navmap_evac_counter = (int) fpgu_strtol(buffer); 
                }
				
            }
            else if (in_env){
            if(in_env_outputting_time){
              
                    env_outputting_time = (double) fpgu_strtod(buffer);
                    
                    set_outputting_time(&env_outputting_time);
                  
              }
            if(in_env_outputting_time_interval){
              
                    env_outputting_time_interval = (double) fpgu_strtod(buffer);
                    
                    set_outputting_time_interval(&env_outputting_time_interval);
                  
              }
            if(in_env_xmin){
              
                    env_xmin = (double) fpgu_strtod(buffer);
                    
                    set_xmin(&env_xmin);
                  
              }
            if(in_env_xmax){
              
                    env_xmax = (double) fpgu_strtod(buffer);
                    
                    set_xmax(&env_xmax);
                  
              }
            if(in_env_ymin){
              
                    env_ymin = (double) fpgu_strtod(buffer);
                    
                    set_ymin(&env_ymin);
                  
              }
            if(in_env_ymax){
              
                    env_ymax = (double) fpgu_strtod(buffer);
                    
                    set_ymax(&env_ymax);
                  
              }
            if(in_env_dt_ped){
              
                    env_dt_ped = (double) fpgu_strtod(buffer);
                    
                    set_dt_ped(&env_dt_ped);
                  
              }
            if(in_env_dt_flood){
              
                    env_dt_flood = (double) fpgu_strtod(buffer);
                    
                    set_dt_flood(&env_dt_flood);
                  
              }
            if(in_env_dt){
              
                    env_dt = (double) fpgu_strtod(buffer);
                    
                    set_dt(&env_dt);
                  
              }
            if(in_env_auto_dt_on){
              
                    env_auto_dt_on = (int) fpgu_strtol(buffer);
                    
                    set_auto_dt_on(&env_auto_dt_on);
                  
              }
            if(in_env_body_as_obstacle_on){
              
                    env_body_as_obstacle_on = (int) fpgu_strtol(buffer);
                    
                    set_body_as_obstacle_on(&env_body_as_obstacle_on);
                  
              }
            if(in_env_ped_roughness_effect_on){
              
                    env_ped_roughness_effect_on = (int) fpgu_strtol(buffer);
                    
                    set_ped_roughness_effect_on(&env_ped_roughness_effect_on);
                  
              }
            if(in_env_body_height){
              
                    env_body_height = (float) fgpu_atof(buffer);
                    
                    set_body_height(&env_body_height);
                  
              }
            if(in_env_init_speed){
              
                    env_init_speed = (float) fgpu_atof(buffer);
                    
                    set_init_speed(&env_init_speed);
                  
              }
            if(in_env_brisk_speed){
              
                    env_brisk_speed = (float) fgpu_atof(buffer);
                    
                    set_brisk_speed(&env_brisk_speed);
                  
              }
            if(in_env_sim_time){
              
                    env_sim_time = (double) fpgu_strtod(buffer);
                    
                    set_sim_time(&env_sim_time);
                  
              }
            if(in_env_DXL){
              
                    env_DXL = (double) fpgu_strtod(buffer);
                    
                    set_DXL(&env_DXL);
                  
              }
            if(in_env_DYL){
              
                    env_DYL = (double) fpgu_strtod(buffer);
                    
                    set_DYL(&env_DYL);
                  
              }
            if(in_env_inflow_start_time){
              
                    env_inflow_start_time = (double) fpgu_strtod(buffer);
                    
                    set_inflow_start_time(&env_inflow_start_time);
                  
              }
            if(in_env_inflow_peak_time){
              
                    env_inflow_peak_time = (double) fpgu_strtod(buffer);
                    
                    set_inflow_peak_time(&env_inflow_peak_time);
                  
              }
            if(in_env_inflow_end_time){
              
                    env_inflow_end_time = (double) fpgu_strtod(buffer);
                    
                    set_inflow_end_time(&env_inflow_end_time);
                  
              }
            if(in_env_inflow_initial_discharge){
              
                    env_inflow_initial_discharge = (double) fpgu_strtod(buffer);
                    
                    set_inflow_initial_discharge(&env_inflow_initial_discharge);
                  
              }
            if(in_env_inflow_peak_discharge){
              
                    env_inflow_peak_discharge = (double) fpgu_strtod(buffer);
                    
                    set_inflow_peak_discharge(&env_inflow_peak_discharge);
                  
              }
            if(in_env_inflow_end_discharge){
              
                    env_inflow_end_discharge = (double) fpgu_strtod(buffer);
                    
                    set_inflow_end_discharge(&env_inflow_end_discharge);
                  
              }
            if(in_env_INFLOW_BOUNDARY){
              
                    env_INFLOW_BOUNDARY = (int) fpgu_strtol(buffer);
                    
                    set_INFLOW_BOUNDARY(&env_INFLOW_BOUNDARY);
                  
              }
            if(in_env_BOUNDARY_EAST_STATUS){
              
                    env_BOUNDARY_EAST_STATUS = (int) fpgu_strtol(buffer);
                    
                    set_BOUNDARY_EAST_STATUS(&env_BOUNDARY_EAST_STATUS);
                  
              }
            if(in_env_BOUNDARY_WEST_STATUS){
              
                    env_BOUNDARY_WEST_STATUS = (int) fpgu_strtol(buffer);
                    
                    set_BOUNDARY_WEST_STATUS(&env_BOUNDARY_WEST_STATUS);
                  
              }
            if(in_env_BOUNDARY_NORTH_STATUS){
              
                    env_BOUNDARY_NORTH_STATUS = (int) fpgu_strtol(buffer);
                    
                    set_BOUNDARY_NORTH_STATUS(&env_BOUNDARY_NORTH_STATUS);
                  
              }
            if(in_env_BOUNDARY_SOUTH_STATUS){
              
                    env_BOUNDARY_SOUTH_STATUS = (int) fpgu_strtol(buffer);
                    
                    set_BOUNDARY_SOUTH_STATUS(&env_BOUNDARY_SOUTH_STATUS);
                  
              }
            if(in_env_x1_boundary){
              
                    env_x1_boundary = (double) fpgu_strtod(buffer);
                    
                    set_x1_boundary(&env_x1_boundary);
                  
              }
            if(in_env_x2_boundary){
              
                    env_x2_boundary = (double) fpgu_strtod(buffer);
                    
                    set_x2_boundary(&env_x2_boundary);
                  
              }
            if(in_env_y1_boundary){
              
                    env_y1_boundary = (double) fpgu_strtod(buffer);
                    
                    set_y1_boundary(&env_y1_boundary);
                  
              }
            if(in_env_y2_boundary){
              
                    env_y2_boundary = (double) fpgu_strtod(buffer);
                    
                    set_y2_boundary(&env_y2_boundary);
                  
              }
            if(in_env_init_depth_boundary){
              
                    env_init_depth_boundary = (double) fpgu_strtod(buffer);
                    
                    set_init_depth_boundary(&env_init_depth_boundary);
                  
              }
            if(in_env_evacuation_on){
              
                    env_evacuation_on = (int) fpgu_strtol(buffer);
                    
                    set_evacuation_on(&env_evacuation_on);
                  
              }
            if(in_env_walking_speed_reduction_in_water_on){
              
                    env_walking_speed_reduction_in_water_on = (int) fpgu_strtol(buffer);
                    
                    set_walking_speed_reduction_in_water_on(&env_walking_speed_reduction_in_water_on);
                  
              }
            if(in_env_freeze_while_instable_on){
              
                    env_freeze_while_instable_on = (int) fpgu_strtol(buffer);
                    
                    set_freeze_while_instable_on(&env_freeze_while_instable_on);
                  
              }
            if(in_env_evacuation_end_time){
              
                    env_evacuation_end_time = (double) fpgu_strtod(buffer);
                    
                    set_evacuation_end_time(&env_evacuation_end_time);
                  
              }
            if(in_env_evacuation_start_time){
              
                    env_evacuation_start_time = (double) fpgu_strtod(buffer);
                    
                    set_evacuation_start_time(&env_evacuation_start_time);
                  
              }
            if(in_env_emergency_exit_number){
              
                    env_emergency_exit_number = (int) fpgu_strtol(buffer);
                    
                    set_emergency_exit_number(&env_emergency_exit_number);
                  
              }
            if(in_env_emer_alarm){
              
                    env_emer_alarm = (int) fpgu_strtol(buffer);
                    
                    set_emer_alarm(&env_emer_alarm);
                  
              }
            if(in_env_HR){
              
                    env_HR = (double) fpgu_strtod(buffer);
                    
                    set_HR(&env_HR);
                  
              }
            if(in_env_max_at_highest_risk){
              
                    env_max_at_highest_risk = (int) fpgu_strtol(buffer);
                    
                    set_max_at_highest_risk(&env_max_at_highest_risk);
                  
              }
            if(in_env_max_at_low_risk){
              
                    env_max_at_low_risk = (int) fpgu_strtol(buffer);
                    
                    set_max_at_low_risk(&env_max_at_low_risk);
                  
              }
            if(in_env_max_at_medium_risk){
              
                    env_max_at_medium_risk = (int) fpgu_strtol(buffer);
                    
                    set_max_at_medium_risk(&env_max_at_medium_risk);
                  
              }
            if(in_env_max_at_high_risk){
              
                    env_max_at_high_risk = (int) fpgu_strtol(buffer);
                    
                    set_max_at_high_risk(&env_max_at_high_risk);
                  
              }
            if(in_env_max_velocity){
              
                    env_max_velocity = (double) fpgu_strtod(buffer);
                    
                    set_max_velocity(&env_max_velocity);
                  
              }
            if(in_env_max_depth){
              
                    env_max_depth = (double) fpgu_strtod(buffer);
                    
                    set_max_depth(&env_max_depth);
                  
              }
            if(in_env_count_population){
              
                    env_count_population = (int) fpgu_strtol(buffer);
                    
                    set_count_population(&env_count_population);
                  
              }
            if(in_env_count_heros){
              
                    env_count_heros = (int) fpgu_strtol(buffer);
                    
                    set_count_heros(&env_count_heros);
                  
              }
            if(in_env_initial_population){
              
                    env_initial_population = (int) fpgu_strtol(buffer);
                    
                    set_initial_population(&env_initial_population);
                  
              }
            if(in_env_evacuated_population){
              
                    env_evacuated_population = (int) fpgu_strtol(buffer);
                    
                    set_evacuated_population(&env_evacuated_population);
                  
              }
            if(in_env_hero_percentage){
              
                    env_hero_percentage = (float) fgpu_atof(buffer);
                    
                    set_hero_percentage(&env_hero_percentage);
                  
              }
            if(in_env_hero_population){
              
                    env_hero_population = (int) fpgu_strtol(buffer);
                    
                    set_hero_population(&env_hero_population);
                  
              }
            if(in_env_sandbagging_on){
              
                    env_sandbagging_on = (int) fpgu_strtol(buffer);
                    
                    set_sandbagging_on(&env_sandbagging_on);
                  
              }
            if(in_env_sandbagging_start_time){
              
                    env_sandbagging_start_time = (double) fpgu_strtod(buffer);
                    
                    set_sandbagging_start_time(&env_sandbagging_start_time);
                  
              }
            if(in_env_sandbagging_end_time){
              
                    env_sandbagging_end_time = (double) fpgu_strtod(buffer);
                    
                    set_sandbagging_end_time(&env_sandbagging_end_time);
                  
              }
            if(in_env_sandbag_length){
              
                    env_sandbag_length = (float) fgpu_atof(buffer);
                    
                    set_sandbag_length(&env_sandbag_length);
                  
              }
            if(in_env_sandbag_height){
              
                    env_sandbag_height = (float) fgpu_atof(buffer);
                    
                    set_sandbag_height(&env_sandbag_height);
                  
              }
            if(in_env_sandbag_width){
              
                    env_sandbag_width = (float) fgpu_atof(buffer);
                    
                    set_sandbag_width(&env_sandbag_width);
                  
              }
            if(in_env_extended_length){
              
                    env_extended_length = (float) fgpu_atof(buffer);
                    
                    set_extended_length(&env_extended_length);
                  
              }
            if(in_env_sandbag_layers){
              
                    env_sandbag_layers = (int) fpgu_strtol(buffer);
                    
                    set_sandbag_layers(&env_sandbag_layers);
                  
              }
            if(in_env_update_stopper){
              
                    env_update_stopper = (int) fpgu_strtol(buffer);
                    
                    set_update_stopper(&env_update_stopper);
                  
              }
            if(in_env_dike_length){
              
                    env_dike_length = (float) fgpu_atof(buffer);
                    
                    set_dike_length(&env_dike_length);
                  
              }
            if(in_env_dike_height){
              
                    env_dike_height = (float) fgpu_atof(buffer);
                    
                    set_dike_height(&env_dike_height);
                  
              }
            if(in_env_dike_width){
              
                    env_dike_width = (float) fgpu_atof(buffer);
                    
                    set_dike_width(&env_dike_width);
                  
              }
            if(in_env_fill_cap){
              
                    env_fill_cap = (int) fpgu_strtol(buffer);
                    
                    set_fill_cap(&env_fill_cap);
                  
              }
            if(in_env_pickup_point){
              
                    env_pickup_point = (int) fpgu_strtol(buffer);
                    
                    set_pickup_point(&env_pickup_point);
                  
              }
            if(in_env_drop_point){
              
                    env_drop_point = (int) fpgu_strtol(buffer);
                    
                    set_drop_point(&env_drop_point);
                  
              }
            if(in_env_pickup_duration){
              
                    env_pickup_duration = (float) fgpu_atof(buffer);
                    
                    set_pickup_duration(&env_pickup_duration);
                  
              }
            if(in_env_drop_duration){
              
                    env_drop_duration = (float) fgpu_atof(buffer);
                    
                    set_drop_duration(&env_drop_duration);
                  
              }
            if(in_env_EMMISION_RATE_EXIT1){
              
                    env_EMMISION_RATE_EXIT1 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT1(&env_EMMISION_RATE_EXIT1);
                  
              }
            if(in_env_EMMISION_RATE_EXIT2){
              
                    env_EMMISION_RATE_EXIT2 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT2(&env_EMMISION_RATE_EXIT2);
                  
              }
            if(in_env_EMMISION_RATE_EXIT3){
              
                    env_EMMISION_RATE_EXIT3 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT3(&env_EMMISION_RATE_EXIT3);
                  
              }
            if(in_env_EMMISION_RATE_EXIT4){
              
                    env_EMMISION_RATE_EXIT4 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT4(&env_EMMISION_RATE_EXIT4);
                  
              }
            if(in_env_EMMISION_RATE_EXIT5){
              
                    env_EMMISION_RATE_EXIT5 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT5(&env_EMMISION_RATE_EXIT5);
                  
              }
            if(in_env_EMMISION_RATE_EXIT6){
              
                    env_EMMISION_RATE_EXIT6 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT6(&env_EMMISION_RATE_EXIT6);
                  
              }
            if(in_env_EMMISION_RATE_EXIT7){
              
                    env_EMMISION_RATE_EXIT7 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT7(&env_EMMISION_RATE_EXIT7);
                  
              }
            if(in_env_EMMISION_RATE_EXIT8){
              
                    env_EMMISION_RATE_EXIT8 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT8(&env_EMMISION_RATE_EXIT8);
                  
              }
            if(in_env_EMMISION_RATE_EXIT9){
              
                    env_EMMISION_RATE_EXIT9 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT9(&env_EMMISION_RATE_EXIT9);
                  
              }
            if(in_env_EMMISION_RATE_EXIT10){
              
                    env_EMMISION_RATE_EXIT10 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT10(&env_EMMISION_RATE_EXIT10);
                  
              }
            if(in_env_EXIT1_PROBABILITY){
              
                    env_EXIT1_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT1_PROBABILITY(&env_EXIT1_PROBABILITY);
                  
              }
            if(in_env_EXIT2_PROBABILITY){
              
                    env_EXIT2_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT2_PROBABILITY(&env_EXIT2_PROBABILITY);
                  
              }
            if(in_env_EXIT3_PROBABILITY){
              
                    env_EXIT3_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT3_PROBABILITY(&env_EXIT3_PROBABILITY);
                  
              }
            if(in_env_EXIT4_PROBABILITY){
              
                    env_EXIT4_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT4_PROBABILITY(&env_EXIT4_PROBABILITY);
                  
              }
            if(in_env_EXIT5_PROBABILITY){
              
                    env_EXIT5_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT5_PROBABILITY(&env_EXIT5_PROBABILITY);
                  
              }
            if(in_env_EXIT6_PROBABILITY){
              
                    env_EXIT6_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT6_PROBABILITY(&env_EXIT6_PROBABILITY);
                  
              }
            if(in_env_EXIT7_PROBABILITY){
              
                    env_EXIT7_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT7_PROBABILITY(&env_EXIT7_PROBABILITY);
                  
              }
            if(in_env_EXIT8_PROBABILITY){
              
                    env_EXIT8_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT8_PROBABILITY(&env_EXIT8_PROBABILITY);
                  
              }
            if(in_env_EXIT9_PROBABILITY){
              
                    env_EXIT9_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT9_PROBABILITY(&env_EXIT9_PROBABILITY);
                  
              }
            if(in_env_EXIT10_PROBABILITY){
              
                    env_EXIT10_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT10_PROBABILITY(&env_EXIT10_PROBABILITY);
                  
              }
            if(in_env_EXIT1_STATE){
              
                    env_EXIT1_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT1_STATE(&env_EXIT1_STATE);
                  
              }
            if(in_env_EXIT2_STATE){
              
                    env_EXIT2_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT2_STATE(&env_EXIT2_STATE);
                  
              }
            if(in_env_EXIT3_STATE){
              
                    env_EXIT3_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT3_STATE(&env_EXIT3_STATE);
                  
              }
            if(in_env_EXIT4_STATE){
              
                    env_EXIT4_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT4_STATE(&env_EXIT4_STATE);
                  
              }
            if(in_env_EXIT5_STATE){
              
                    env_EXIT5_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT5_STATE(&env_EXIT5_STATE);
                  
              }
            if(in_env_EXIT6_STATE){
              
                    env_EXIT6_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT6_STATE(&env_EXIT6_STATE);
                  
              }
            if(in_env_EXIT7_STATE){
              
                    env_EXIT7_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT7_STATE(&env_EXIT7_STATE);
                  
              }
            if(in_env_EXIT8_STATE){
              
                    env_EXIT8_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT8_STATE(&env_EXIT8_STATE);
                  
              }
            if(in_env_EXIT9_STATE){
              
                    env_EXIT9_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT9_STATE(&env_EXIT9_STATE);
                  
              }
            if(in_env_EXIT10_STATE){
              
                    env_EXIT10_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT10_STATE(&env_EXIT10_STATE);
                  
              }
            if(in_env_EXIT1_CELL_COUNT){
              
                    env_EXIT1_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT1_CELL_COUNT(&env_EXIT1_CELL_COUNT);
                  
              }
            if(in_env_EXIT2_CELL_COUNT){
              
                    env_EXIT2_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT2_CELL_COUNT(&env_EXIT2_CELL_COUNT);
                  
              }
            if(in_env_EXIT3_CELL_COUNT){
              
                    env_EXIT3_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT3_CELL_COUNT(&env_EXIT3_CELL_COUNT);
                  
              }
            if(in_env_EXIT4_CELL_COUNT){
              
                    env_EXIT4_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT4_CELL_COUNT(&env_EXIT4_CELL_COUNT);
                  
              }
            if(in_env_EXIT5_CELL_COUNT){
              
                    env_EXIT5_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT5_CELL_COUNT(&env_EXIT5_CELL_COUNT);
                  
              }
            if(in_env_EXIT6_CELL_COUNT){
              
                    env_EXIT6_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT6_CELL_COUNT(&env_EXIT6_CELL_COUNT);
                  
              }
            if(in_env_EXIT7_CELL_COUNT){
              
                    env_EXIT7_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT7_CELL_COUNT(&env_EXIT7_CELL_COUNT);
                  
              }
            if(in_env_EXIT8_CELL_COUNT){
              
                    env_EXIT8_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT8_CELL_COUNT(&env_EXIT8_CELL_COUNT);
                  
              }
            if(in_env_EXIT9_CELL_COUNT){
              
                    env_EXIT9_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT9_CELL_COUNT(&env_EXIT9_CELL_COUNT);
                  
              }
            if(in_env_EXIT10_CELL_COUNT){
              
                    env_EXIT10_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT10_CELL_COUNT(&env_EXIT10_CELL_COUNT);
                  
              }
            if(in_env_TIME_SCALER){
              
                    env_TIME_SCALER = (float) fgpu_atof(buffer);
                    
                    set_TIME_SCALER(&env_TIME_SCALER);
                  
              }
            if(in_env_STEER_WEIGHT){
              
                    env_STEER_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_STEER_WEIGHT(&env_STEER_WEIGHT);
                  
              }
            if(in_env_AVOID_WEIGHT){
              
                    env_AVOID_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_AVOID_WEIGHT(&env_AVOID_WEIGHT);
                  
              }
            if(in_env_COLLISION_WEIGHT){
              
                    env_COLLISION_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_COLLISION_WEIGHT(&env_COLLISION_WEIGHT);
                  
              }
            if(in_env_GOAL_WEIGHT){
              
                    env_GOAL_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_GOAL_WEIGHT(&env_GOAL_WEIGHT);
                  
              }
            if(in_env_PedHeight_60_110_probability){
              
                    env_PedHeight_60_110_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedHeight_60_110_probability(&env_PedHeight_60_110_probability);
                  
              }
            if(in_env_PedHeight_110_140_probability){
              
                    env_PedHeight_110_140_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedHeight_110_140_probability(&env_PedHeight_110_140_probability);
                  
              }
            if(in_env_PedHeight_140_163_probability){
              
                    env_PedHeight_140_163_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedHeight_140_163_probability(&env_PedHeight_140_163_probability);
                  
              }
            if(in_env_PedHeight_163_170_probability){
              
                    env_PedHeight_163_170_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedHeight_163_170_probability(&env_PedHeight_163_170_probability);
                  
              }
            if(in_env_PedHeight_170_186_probability){
              
                    env_PedHeight_170_186_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedHeight_170_186_probability(&env_PedHeight_170_186_probability);
                  
              }
            if(in_env_PedHeight_186_194_probability){
              
                    env_PedHeight_186_194_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedHeight_186_194_probability(&env_PedHeight_186_194_probability);
                  
              }
            if(in_env_PedHeight_194_210_probability){
              
                    env_PedHeight_194_210_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedHeight_194_210_probability(&env_PedHeight_194_210_probability);
                  
              }
            if(in_env_PedAge_10_17_probability){
              
                    env_PedAge_10_17_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedAge_10_17_probability(&env_PedAge_10_17_probability);
                  
              }
            if(in_env_PedAge_18_29_probability){
              
                    env_PedAge_18_29_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedAge_18_29_probability(&env_PedAge_18_29_probability);
                  
              }
            if(in_env_PedAge_30_39_probability){
              
                    env_PedAge_30_39_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedAge_30_39_probability(&env_PedAge_30_39_probability);
                  
              }
            if(in_env_PedAge_40_49_probability){
              
                    env_PedAge_40_49_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedAge_40_49_probability(&env_PedAge_40_49_probability);
                  
              }
            if(in_env_PedAge_50_59_probability){
              
                    env_PedAge_50_59_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedAge_50_59_probability(&env_PedAge_50_59_probability);
                  
              }
            if(in_env_PedAge_60_69_probability){
              
                    env_PedAge_60_69_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedAge_60_69_probability(&env_PedAge_60_69_probability);
                  
              }
            if(in_env_PedAge_70_79_probability){
              
                    env_PedAge_70_79_probability = (int) fpgu_strtol(buffer);
                    
                    set_PedAge_70_79_probability(&env_PedAge_70_79_probability);
                  
              }
            if(in_env_excluded_age_probability){
              
                    env_excluded_age_probability = (int) fpgu_strtol(buffer);
                    
                    set_excluded_age_probability(&env_excluded_age_probability);
                  
              }
            if(in_env_gender_female_probability){
              
                    env_gender_female_probability = (int) fpgu_strtol(buffer);
                    
                    set_gender_female_probability(&env_gender_female_probability);
                  
              }
            if(in_env_gender_male_probability){
              
                    env_gender_male_probability = (int) fpgu_strtol(buffer);
                    
                    set_gender_male_probability(&env_gender_male_probability);
                  
              }
            if(in_env_SCALE_FACTOR){
              
                    env_SCALE_FACTOR = (float) fgpu_atof(buffer);
                    
                    set_SCALE_FACTOR(&env_SCALE_FACTOR);
                  
              }
            if(in_env_I_SCALER){
              
                    env_I_SCALER = (float) fgpu_atof(buffer);
                    
                    set_I_SCALER(&env_I_SCALER);
                  
              }
            if(in_env_MIN_DISTANCE){
              
                    env_MIN_DISTANCE = (float) fgpu_atof(buffer);
                    
                    set_MIN_DISTANCE(&env_MIN_DISTANCE);
                  
              }
            if(in_env_excitement_on){
              
                    env_excitement_on = (int) fpgu_strtol(buffer);
                    
                    set_excitement_on(&env_excitement_on);
                  
              }
            if(in_env_walk_run_switch){
              
                    env_walk_run_switch = (int) fpgu_strtol(buffer);
                    
                    set_walk_run_switch(&env_walk_run_switch);
                  
              }
            if(in_env_preoccupying_on){
              
                    env_preoccupying_on = (int) fpgu_strtol(buffer);
                    
                    set_preoccupying_on(&env_preoccupying_on);
                  
              }
            if(in_env_poly_hydrograph_on){
              
                    env_poly_hydrograph_on = (int) fpgu_strtol(buffer);
                    
                    set_poly_hydrograph_on(&env_poly_hydrograph_on);
                  
              }
            if(in_env_stop_emission_on){
              
                    env_stop_emission_on = (int) fpgu_strtol(buffer);
                    
                    set_stop_emission_on(&env_stop_emission_on);
                  
              }
            if(in_env_goto_emergency_exit_on){
              
                    env_goto_emergency_exit_on = (int) fpgu_strtol(buffer);
                    
                    set_goto_emergency_exit_on(&env_goto_emergency_exit_on);
                  
              }
            if(in_env_escape_route_finder_on){
              
                    env_escape_route_finder_on = (int) fpgu_strtol(buffer);
                    
                    set_escape_route_finder_on(&env_escape_route_finder_on);
                  
              }
            if(in_env_dir_times){
              
                    env_dir_times = (int) fpgu_strtol(buffer);
                    
                    set_dir_times(&env_dir_times);
                  
              }
            if(in_env_no_return_on){
              
                    env_no_return_on = (int) fpgu_strtol(buffer);
                    
                    set_no_return_on(&env_no_return_on);
                  
              }
            if(in_env_wdepth_perc_thresh){
              
                    env_wdepth_perc_thresh = (float) fgpu_atof(buffer);
                    
                    set_wdepth_perc_thresh(&env_wdepth_perc_thresh);
                  
              }
            if(in_env_follow_popular_exit_on){
              
                    env_follow_popular_exit_on = (int) fpgu_strtol(buffer);
                    
                    set_follow_popular_exit_on(&env_follow_popular_exit_on);
                  
              }
            if(in_env_popular_exit){
              
                    env_popular_exit = (int) fpgu_strtol(buffer);
                    
                    set_popular_exit(&env_popular_exit);
                  
              }
            
            }
		/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
            // Check if we are a comment, when we are in a tag and buffer[0:2] == "!--"
            if(i == 2 && c == '-' && buffer[1] == '-' && buffer[0] == '!'){
                in_comment = 1;
                // Reset the buffer and i.
                buffer[0] = 0;
                i = 0;
            }

            // Store the character and increment the counter
            buffer[i] = c;
            i++;

		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
    // If no bytes were read, raise a warning.
    if(bytesRead == 0){
        fprintf(stdout, "Warning: %s is an empty file\n", inputpath);
        fflush(stdout);
    }

    // If the in_comment flag is still marked, issue a warning.
    if(in_comment){
        fprintf(stdout, "Warning: Un-terminated comment in %s\n", inputpath);
        fflush(stdout);
    }    

	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}


/* Methods to load static networks from disk */
