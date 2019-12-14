/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#include "utils.h"
#include "canny_edge_host.h"
#include "canny_edge_device.h"

#define INPUT_FILES_PATH "../../../Input_images/"
#define OUTPUT_FILES_PATH "../../../Output_images/"
#define OUTPUT_CUDA_FILES_PATH "../../../Output_images_cuda/"
// only .png files are supported!
#define INPUT_FILES_EXTENSIONS ".png"

//daimler_800_777, bmw_1683_1230, range_rover_1920_1080, car_1920_1080, Red_Mazda_2528_1368, audi_2913_1539, Horses_Run_Animals_horse_9192x6012, Valve_original_paper_640_480
// WARNING - Horses_Run_Animals_horse_9192x6012 is not present in our repository, so it will fail!
#define INPUT_FILE_NAME "Red_Mazda_2528_1368"

#define OUTPUT_GAUSSIATED_FILE_NAME "gaussiated"
#define OUTPUT_SOBELED_GRAD_X_FILE_NAME "sobeled_grad_x"
#define OUTPUT_SOBELED_GRAD_Y_FILE_NAME "sobeled_grad_y"
#define OUTPUT_NON_MAX_SUPPRESSED_FILE_NAME "nms"
#define OUTPUT_DOUBLE_THRESHOLDED_FILE_NAME "double_thresholded"
#define OUTPUT_EDGE_TRACKED_FILE_NAME "edge_tracked"

#define OUTPUT_CUDA_GAUSSIATED_FILE_NAME "gaussiated_cuda"
#define OUTPUT_CUDA_SOBELED_GRAD_X_FILE_NAME "sobeled_grad_x_cuda"
#define OUTPUT_CUDA_SOBELED_GRAD_Y_FILE_NAME "sobeled_grad_y_cuda"
#define OUTPUT_CUDA_NON_MAX_SUPPRESSED_FILE_NAME "nms_cuda"
#define OUTPUT_CUDA_DOUBLE_THRESHOLDED_FILE_NAME "double_thresholded_cuda"
#define OUTPUT_CUDA_EDGE_TRACKED_FILE_NAME "edge_tracked_cuda"

// writes output image after every stage to log file
//#define DEBUG

// dumps only final image
//#define ONLY_FINAL

int main(int argc, char **argv) {

#ifdef DEBUG
	char file_name[50];
	sprintf(file_name, "../../../Output_images/log_host.txt");

	FILE *f = fopen(file_name, "w");
#endif //DEBUG

	char input_file_name[200];
	char output_file_name[200];
	sprintf(input_file_name, "%s%s%s", INPUT_FILES_PATH, INPUT_FILE_NAME, INPUT_FILES_EXTENSIONS);

	// read image from disk
	// SILENTLY FAILS WITHOUT ERROR!
	CByteImage cbimage;
	ReadImage(cbimage, input_file_name);

	// convert to grayscale
	CByteImage cbimage_gray = ConvertToGray(cbimage);
	CShape cbimage_gray_shape = cbimage_gray.Shape();

	// convert the image to a regular array
	gray_image gimage = gray_image(convert_CByteImage_to_array(cbimage_gray), \
								   cbimage_gray_shape.width, 				  \
								   cbimage_gray_shape.height);

	printf("Successfully loaded %s of height:%d, width:%d\n\n\n", input_file_name, gimage.get_height(), gimage.get_width());

	canny_edge_host from_host = canny_edge_host(gimage.get_host_gimage(), gimage.get_width(), gimage.get_height());

#ifdef DEBUG
	//fprintf(f, "image %s\n", INPUT_FILE_NAME);
	//print_log_matrix(f, gimage.get_host_gimage(), from_host.get_width(), from_host.get_height());

	fprintf(f, "gaussian_kernel\n");
	print_log_matrix(f, from_host.get_gaussian_kernel(), GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE);

	fprintf(f, "sobel_filter_x\n");
	print_log_matrix(f, from_host.get_sobel_filter_x(), SOBEL_FILTER_SIZE, SOBEL_FILTER_SIZE);
	fprintf(f, "sobel_filter_y\n");
	print_log_matrix(f, from_host.get_sobel_filter_y(), SOBEL_FILTER_SIZE, SOBEL_FILTER_SIZE);
//	from_host.print_gaussian_kernel();
//	from_host.print_sobel_filters();
#endif //DEBUG

	// ############################################################### CPU Step 1 ############################################################### //
	from_host.apply_gaussian_kernel();
#ifdef DEBUG
	fprintf(f, "gaussiated_image\n");
	print_log_matrix(f, from_host.get_gaussiated_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_FILES_PATH, INPUT_FILE_NAME, OUTPUT_GAUSSIATED_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_host.get_gaussiated_image(), from_host.get_width(), from_host.get_height(), output_file_name, false);
#endif //ONLY_FINAL

	// ############################################################### CPU Step 2 ############################################################### //
	from_host.compute_pixel_thresholds();

	// ############################################################### CPU Step 3 ############################################################### //
	from_host.apply_sobel_filter_x();
#ifdef DEBUG
	fprintf(f, "sobeled_grad_x_image\n");
	print_log_matrix(f, from_host.get_sobeled_grad_x_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_FILES_PATH, INPUT_FILE_NAME, OUTPUT_SOBELED_GRAD_X_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_host.get_sobeled_grad_x_image(), from_host.get_width(), from_host.get_height(), output_file_name, false);
#endif //ONLY_FINAL

	// ############################################################### CPU Step 4 ############################################################### //
	from_host.apply_sobel_filter_y();
#ifdef DEBUG
	fprintf(f, "sobeled_grad_y_image\n");
	print_log_matrix(f, from_host.get_sobeled_grad_y_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_FILES_PATH, INPUT_FILE_NAME, OUTPUT_SOBELED_GRAD_Y_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_host.get_sobeled_grad_y_image(), from_host.get_width(), from_host.get_height(), output_file_name, false);
#endif //ONLY_FINAL

	// ############################################################### CPU Step 5 ############################################################### //
	from_host.calculate_sobel_magnitude();
#ifdef DEBUG
	fprintf(f, "sobeled_mag_image\n");
	print_log_matrix(f, from_host.get_sobeled_mag_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	//write_image_to_file(from_host.get_gaussiated_image(), from_host.get_width(), from_host.get_height(), OUTPUT_GAUSSIATED_FILE_NAME, false);

	// ############################################################### CPU Step 6 ############################################################### //
	from_host.calculate_sobel_direction();
#ifdef DEBUG
	fprintf(f, "sobeled_dir_image\n");
	print_log_matrix(f, from_host.get_sobeled_dir_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG

	// ############################################################### CPU Step 7 ############################################################### //
	from_host.apply_non_max_suppression();
#ifdef DEBUG
	fprintf(f, "non_max_suppressed_image\n");
	print_log_matrix(f, from_host.get_non_max_suppressed_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_FILES_PATH, INPUT_FILE_NAME, OUTPUT_NON_MAX_SUPPRESSED_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_host.get_non_max_suppressed_image(), from_host.get_width(), from_host.get_height(), output_file_name, false);
#endif //ONLY_FINAL

	// ############################################################### CPU Step 8 ############################################################### //
	from_host.apply_double_thresholds();
#ifdef DEBUG
	fprintf(f, "double_thresholded_image\n");
	print_log_matrix(f, from_host.get_double_thresholded_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_FILES_PATH, INPUT_FILE_NAME, OUTPUT_DOUBLE_THRESHOLDED_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_host.get_double_thresholded_image(), from_host.get_width(), from_host.get_height(), output_file_name, false);
#endif //ONLY_FINAL

	// ############################################################### CPU Step 9 ############################################################### //
	from_host.apply_hysteresis_edge_tracking();
#ifdef DEBUG
	fprintf(f, "edge_tracked_image\n");
	print_log_matrix(f, from_host.get_edge_tracked_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_FILES_PATH, INPUT_FILE_NAME, OUTPUT_EDGE_TRACKED_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_host.get_edge_tracked_image(), from_host.get_width(), from_host.get_height(), output_file_name, false);

	printf("\nCPU took %.2fms\n\n\n", from_host.get_total_time_taken());

#ifdef DEBUG
	fclose(f);
#endif //DEBUG

	// ############################################################### GPU Start ############################################################### //
#ifdef DEBUG
	sprintf(file_name, "../../../Output_images_cuda/log_device.txt");
	f = fopen(file_name, "w");
#endif //DEBUG

	canny_edge_device from_device = canny_edge_device(gimage.get_host_gimage(), gimage.get_width(), gimage.get_height());

#ifdef DEBUG
	fprintf(f, "gaussian_kernel_cuda\n");
	float *gaussian_kernel_temp = (float*)malloc(sizeof(float) * GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE);
	CHECK(cudaMemcpy(gaussian_kernel_temp, from_device.get_gaussian_kernel(), sizeof(float) * GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE, cudaMemcpyDeviceToHost));
	print_log_matrix(f, gaussian_kernel_temp, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE);
	free(gaussian_kernel_temp);

	fprintf(f, "sobel_filter_x_cuda\n");
	float *sobel_filter_x_temp = (float*)malloc(sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE);
	CHECK(cudaMemcpy(sobel_filter_x_temp, from_device.get_sobel_filter_x(), sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE, cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobel_filter_x_temp, SOBEL_FILTER_SIZE, SOBEL_FILTER_SIZE);
	free(sobel_filter_x_temp);

	fprintf(f, "sobel_filter_y_cuda\n");
	float *sobel_filter_y_temp = (float*)malloc(sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE);
	CHECK(cudaMemcpy(sobel_filter_y_temp, from_device.get_sobel_filter_y(), sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE, cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobel_filter_y_temp, SOBEL_FILTER_SIZE, SOBEL_FILTER_SIZE);
	free(sobel_filter_y_temp);
#endif //DEBUG

	// ############################################################### GPU Step 1 ############################################################### //

	from_device.apply_gaussian_kernel();
#ifdef DEBUG
	fprintf(f, "gaussiated_image_cuda\n");
	float *gaussiated_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(gaussiated_image_temp, from_device.get_gaussiated_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, gaussiated_image_temp, from_device.get_width(), from_device.get_height());
	free(gaussiated_image_temp);
#endif //DEBUG

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_CUDA_FILES_PATH, INPUT_FILE_NAME, OUTPUT_CUDA_GAUSSIATED_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_device.get_gaussiated_image(), from_device.get_width(), from_device.get_height(), output_file_name, true);
#endif //ONLY_FINAL

	// ############################################################### GPU Step 2 ############################################################### //

	from_device.compute_pixel_thresholds();

	// ############################################################### GPU Step 3,4 ############################################################### //

	from_device.streamed_apply_sobel_filter_x_y();
	
	//from_device.apply_sobel_filter_x();
	//CHECK(cudaMemcpy(from_device.get_sobeled_grad_x_image(), from_host.get_sobeled_grad_x_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyHostToDevice));
#ifdef DEBUG
	fprintf(f, "sobeled_grad_x_image_cuda\n");
	float *sobeled_grad_x_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(sobeled_grad_x_image_temp, from_device.get_sobeled_grad_x_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobeled_grad_x_image_temp, from_device.get_width(), from_device.get_height());
	free(sobeled_grad_x_image_temp);
#endif //DEBUG

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_CUDA_FILES_PATH, INPUT_FILE_NAME, OUTPUT_CUDA_SOBELED_GRAD_X_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_device.get_sobeled_grad_x_image(), from_device.get_width(), from_device.get_height(), output_file_name, true);
#endif //ONLY_FINAL

	//from_device.apply_sobel_filter_y();
#ifdef DEBUG
	fprintf(f, "sobeled_grad_y_image_cuda\n");
	float *sobeled_grad_y_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(sobeled_grad_y_image_temp, from_device.get_sobeled_grad_y_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobeled_grad_y_image_temp, from_device.get_width(), from_device.get_height());
	free(sobeled_grad_y_image_temp);
#endif //DEBUG

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_CUDA_FILES_PATH, INPUT_FILE_NAME, OUTPUT_CUDA_SOBELED_GRAD_Y_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_device.get_sobeled_grad_y_image(), from_device.get_width(), from_device.get_height(), output_file_name, true);
#endif //ONLY_FINAL

	// ############################################################### GPU Step 5,6 ############################################################### //

	from_device.streamed_calculate_sobel_magnitude_direction();
	//from_device.calculate_sobel_magnitude();
#ifdef DEBUG
	fprintf(f, "sobeled_mag_image_cuda\n");
	float *sobeled_mag_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(sobeled_mag_image_temp, from_device.get_sobeled_mag_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobeled_mag_image_temp, from_device.get_width(), from_device.get_height());
	free(sobeled_mag_image_temp);
#endif //DEBUG

	//from_device.calculate_sobel_direction();
#ifdef DEBUG
	fprintf(f, "sobeled_dir_image_cuda\n");
	float *sobeled_dir_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(sobeled_dir_image_temp, from_device.get_sobeled_dir_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobeled_dir_image_temp, from_device.get_width(), from_device.get_height());
	free(sobeled_dir_image_temp);
#endif //DEBUG

	// ############################################################### GPU Step 7 ############################################################### //

	from_device.apply_non_max_suppression();
#ifdef DEBUG
	fprintf(f, "non_max_suppressed_image_cuda\n");
	float *non_max_suppressed_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(non_max_suppressed_image_temp, from_device.get_non_max_suppressed_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, non_max_suppressed_image_temp, from_device.get_width(), from_device.get_height());
	free(non_max_suppressed_image_temp);
#endif //DEBUG

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_CUDA_FILES_PATH, INPUT_FILE_NAME, OUTPUT_CUDA_NON_MAX_SUPPRESSED_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_device.get_non_max_suppressed_image(), from_device.get_width(), from_device.get_height(), output_file_name, true);
#endif //ONLY_FINAL

	//from_device.compute_pixel_thresholds();

	// ############################################################### GPU Step 8 ############################################################### //

	from_device.apply_double_thresholds();
#ifdef DEBUG
	fprintf(f, "double_thresholded_image_cuda\n");
	float *double_thresholded_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(double_thresholded_image_temp, from_device.get_double_thresholded_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, double_thresholded_image_temp, from_device.get_width(), from_device.get_height());
	free(double_thresholded_image_temp);
#endif //DEBUG'

#ifndef ONLY_FINAL
	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_CUDA_FILES_PATH, INPUT_FILE_NAME, OUTPUT_CUDA_DOUBLE_THRESHOLDED_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_device.get_double_thresholded_image(), from_device.get_width(), from_device.get_height(), output_file_name, true);
#endif //ONLY_FINAL

	// ############################################################### GPU Step 9 ############################################################### //

	from_device.apply_hysteresis_edge_tracking();
#ifdef DEBUG
	fprintf(f, "edge_tracked_image_cuda\n");
	float *edge_tracked_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(edge_tracked_image_temp, from_device.get_edge_tracked_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, edge_tracked_image_temp, from_device.get_width(), from_device.get_height());
	free(edge_tracked_image_temp);
#endif //DEBUG

	sprintf(output_file_name, "%s%s_%s%s", OUTPUT_CUDA_FILES_PATH, INPUT_FILE_NAME, OUTPUT_CUDA_EDGE_TRACKED_FILE_NAME, INPUT_FILES_EXTENSIONS);
	write_image_to_file(from_device.get_edge_tracked_image(), from_device.get_width(), from_device.get_height(), output_file_name, true);

	printf("\nCUDA took %.2fms\n\n", from_device.get_total_time_taken());

#ifdef DEBUG
	fclose(f);
#endif //DEBUG

}