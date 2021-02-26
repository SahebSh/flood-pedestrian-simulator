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
#include <glm/glm.hpp>
#include "FloodMap.h"
#include "BufferObjects.h"


/** Macro for toggling drawing of the navigation map arrows */
int drawFloodMap  = 1;
bool view_water = true;

// bo variables
GLuint sphereVerts;
GLuint sphereNormals;

//Simulation output buffers/textures
cudaGraphicsResource_t FloodCell_Default_cgr;
GLuint FloodCell_Default_tbo;
GLuint FloodCell_Default_displacementTex;


// vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
GLuint vs_displacementMap;
GLuint vs_mapIndex;
GLuint vs_water;
GLuint vs_NM_WIDTH;
GLuint vs_ENV_MAX;
GLuint vs_ENV_WIDTH;

int initGL();
void initShader();
void setVertexBufferData();
static void setSphereVertex(glm::vec3*, int, int);

//external prototypes imported from FLAME GPU
extern int get_agent_FloodCell_MAX_count();
extern int get_agent_FloodCell_Default_count();


void initFloodMap()
{
	//init shader
	initShader();

	// create VBO's
	createVBO(&sphereVerts, GL_ARRAY_BUFFER, SPHERE_SLICES* (SPHERE_STACKS + 1) * sizeof(glm::vec3));
	createVBO(&sphereNormals, GL_ARRAY_BUFFER, SPHERE_SLICES* (SPHERE_STACKS + 1) * sizeof(glm::vec3));
	setVertexBufferData();

	// create TBO
	createTBO(&FloodCell_Default_tbo, &FloodCell_Default_displacementTex, get_agent_FloodCell_MAX_count() * sizeof(glm::vec4));
	//register graphics
	registerBO(&FloodCell_Default_cgr, &FloodCell_Default_tbo);


}


void renderFloodMap()
{	
	generate_instances(&FloodCell_Default_cgr);

	//bind vertex program
	glUseProgram(shaderProgram);

	// Draw FloodCell Agents in Default state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, FloodCell_Default_displacementTex);

	//set water or h0 mode
	glUniform1i(vs_water, view_water);

	//loop
	for (int i = 0; i< get_agent_FloodCell_Default_count(); i++) {
		glVertexAttrib1f(vs_mapIndex, (float)i);

		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS + 1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	glUseProgram(0);
	
}

void setFloodMapDisplayMode(FLOODMAP_VIEW_MODE vm) {

	if (vm == FLOODMAP_VIEW_WATER) {
		view_water = true;
	}
	else if (vm == FLOODMAP_VIEW_Z0) {
		view_water = false;
	}

}



void initShader()
{
	const char* v = floodmap_vshader_source;

	//vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &v, 0);
	glCompileShader(vertexShader);


	//program
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glLinkProgram(shaderProgram);

	// check for errors
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(vertexShader, 262144, &len, data);
		printf("%s", data);
	}

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Program Link Error\n");
	}

	//set uniforms (need to use prgram to do so)
	glUseProgram(shaderProgram);

	// get shader variables
	vs_displacementMap = glGetUniformLocation(shaderProgram, "displacementMap");
	vs_mapIndex = glGetAttribLocation(shaderProgram, "mapIndex");
	vs_water = glGetUniformLocation(shaderProgram, "water");
	vs_NM_WIDTH = glGetUniformLocation(shaderProgram, "FM_WIDTH");
	vs_ENV_MAX = glGetUniformLocation(shaderProgram, "ENV_MAX");
	vs_ENV_WIDTH = glGetUniformLocation(shaderProgram, "ENV_WIDTH");

	int fm_width = (int)floor(sqrt((float)get_agent_FloodCell_MAX_count()));
	glUniform1f(vs_NM_WIDTH, (float)fm_width);
	glUniform1f(vs_ENV_MAX, ENV_MAX);
	glUniform1f(vs_ENV_WIDTH, ENV_WIDTH);
	glUseProgram(0);
	checkGLError();
}


static void setSphereNormal(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358;

	double sl = 2 * PI*slice / SPHERE_SLICES;
	double st = 2 * PI*stack / SPHERE_STACKS;

	data->x = cos(st)*sin(sl);
	data->y = sin(st)*sin(sl);
	data->z = cos(sl);
}

void setVertexBufferData()
{
	int slice, stack;
	int i;

	// upload vertex points data
	glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
	glm::vec3* verts = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice = 0; slice<SPHERE_SLICES / 2; slice++) {
		for (stack = 0; stack <= SPHERE_STACKS; stack++) {
			setSphereVertex(&verts[i++], slice, stack);
			setSphereVertex(&verts[i++], slice + 1, stack);
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// upload vertex normal data
	glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
	glm::vec3* normals = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice = 0; slice<SPHERE_SLICES / 2; slice++) {
		for (stack = 0; stack <= SPHERE_STACKS; stack++) {
			setSphereNormal(&normals[i++], slice, stack);
			setSphereNormal(&normals[i++], slice + 1, stack);
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
}



static void setSphereVertex(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358;

	double sl = 2 * PI*slice / SPHERE_SLICES;
	double st = 2 * PI*stack / SPHERE_STACKS;

	data->x = cos(st)*sin(sl) * SPHERE_RADIUS;
	data->y = sin(st)*sin(sl) * SPHERE_RADIUS;
	data->z = cos(sl) * SPHERE_RADIUS;
}


void setFloodMapOnOff(TOGGLE_STATE state)
{
	drawFloodMap = state;
}


void toggleFloodMapOnOff()
{
	drawFloodMap = !drawFloodMap;
}
