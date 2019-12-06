/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

/* 	Implementation of class gray_image
*/
#include "gray_image.h"

/* 	constructor 
*/
gray_image::gray_image(float *h_gimage, int width, int height) {

	assert(width > 0);
	assert(height > 0);

	this->h_gimage = (float *)malloc(sizeof(float) * width * height);
	assert(this->h_gimage != NULL);
	memcpy(this->h_gimage, h_gimage, sizeof(float) * width * height);
	
	this->width = width;
	this->height = height;
}

/* **************************************************************************************************** */

/* 	destructor
*/
gray_image::~gray_image() {
	if (this->h_gimage != NULL) free(h_gimage);
}

/* **************************************************************************************************** */

/* 	getter for h_gimage
*/
float *gray_image::get_host_gimage() {
	return this->h_gimage;
}

/* **************************************************************************************************** */

/* 	getter for gimage width
*/
int gray_image::get_width() {
	return this->width;
}

/* **************************************************************************************************** */

/* 	getter for gimage height
*/
int gray_image::get_height() {
	return this->height;
}

/* **************************************************************************************************** */

/* 	getter for a pixel from h_gimage
*/
int gray_image::get_pixel(int x, int y) {
	return this->h_gimage[y * this->width + x];
}