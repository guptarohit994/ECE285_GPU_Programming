/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#include "utils.h"
#include "canny_edge_host.h"
#include "canny_edge_device.h"

#define INPUT_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Input_images/daimler_800_410.png"//daimler_800_410, mercedes_logo_20_20
#define OUTPUT_GAUSSIATED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_gaussiated.png"
#define OUTPUT_SOBELED_GRAD_X_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_sobeled_grad_x.png"
#define OUTPUT_SOBELED_GRAD_Y_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_sobeled_grad_y.png"
//#define OUTPUT_GAUSSIATED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_gaussiated.png"
#define OUTPUT_NON_MAX_SUPPRESSED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_nms.png"
#define OUTPUT_DOUBLE_THRESHOLDED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_double_thresholded.png"
#define OUTPUT_EDGE_TRACKED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_edge_tracked.png"

#define OUTPUT_CUDA_GAUSSIATED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images_cuda/tree_gaussiated_cuda.png"
#define OUTPUT_CUDA_SOBELED_GRAD_X_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images_cuda/tree_sobeled_grad_x_cuda.png"
#define OUTPUT_CUDA_SOBELED_GRAD_Y_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images_cuda/tree_sobeled_grad_y_cuda.png"
//#define OUTPUT_GAUSSIATED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images_cuda/tree_gaussiated.png"
#define OUTPUT_CUDA_NON_MAX_SUPPRESSED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images_cuda/tree_nms_cuda.png"
#define OUTPUT_CUDA_DOUBLE_THRESHOLDED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images_cuda/tree_double_thresholded_cuda.png"
#define OUTPUT_CUDA_EDGE_TRACKED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images_cuda/tree_edge_tracked_cuda.png"

//#define DEBUG

int main(int argc, char **argv) {

#ifdef DEBUG
	char file_name[50];
	sprintf(file_name, "../../../Output_images/log_host.txt");

	FILE *f = fopen(file_name, "w");
#endif //DEBUG

	// read image from disk
	// SILENTLY FAILS WITHOUT ERROR!
	CByteImage cbimage;
	ReadImage(cbimage, INPUT_FILE_NAME);

	// convert to grayscale
	CByteImage cbimage_gray = ConvertToGray(cbimage);
	CShape cbimage_gray_shape = cbimage_gray.Shape();

	// convert the image to a regular array
	gray_image gimage = gray_image(convert_CByteImage_to_array(cbimage_gray), \
								   cbimage_gray_shape.width, 				  \
								   cbimage_gray_shape.height);

	printf("Successfully loaded %s of height:%d, width:%d\n", INPUT_FILE_NAME, gimage.get_height(), gimage.get_width());

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
#else
	from_host.print_gaussian_kernel();
	from_host.print_sobel_filters();
#endif //DEBUG

	from_host.apply_gaussian_kernel();
#ifdef DEBUG
	fprintf(f, "gaussiated_image\n");
	print_log_matrix(f, from_host.get_gaussiated_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_gaussiated_image(), from_host.get_width(), from_host.get_height(), OUTPUT_GAUSSIATED_FILE_NAME, false);

	//from_host.compute_pixel_thresholds();

	from_host.apply_sobel_filter_x();
#ifdef DEBUG
	fprintf(f, "sobeled_grad_x_image\n");
	print_log_matrix(f, from_host.get_sobeled_grad_x_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_sobeled_grad_x_image(), from_host.get_width(), from_host.get_height(), OUTPUT_SOBELED_GRAD_X_FILE_NAME, false);

	from_host.apply_sobel_filter_y();
#ifdef DEBUG
	fprintf(f, "sobeled_grad_y_image\n");
	print_log_matrix(f, from_host.get_sobeled_grad_y_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_sobeled_grad_y_image(), from_host.get_width(), from_host.get_height(), OUTPUT_SOBELED_GRAD_Y_FILE_NAME, false);

	from_host.calculate_sobel_magnitude();
#ifdef DEBUG
	fprintf(f, "sobeled_mag_image\n");
	print_log_matrix(f, from_host.get_sobeled_mag_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	//write_image_to_file(from_host.get_gaussiated_image(), from_host.get_width(), from_host.get_height(), OUTPUT_GAUSSIATED_FILE_NAME, false);

	from_host.calculate_sobel_direction();
#ifdef DEBUG
	fprintf(f, "sobeled_dir_image\n");
	print_log_matrix(f, from_host.get_sobeled_dir_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG

	from_host.apply_non_max_suppression();
#ifdef DEBUG
	fprintf(f, "non_max_suppressed_image\n");
	print_log_matrix(f, from_host.get_non_max_suppressed_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_non_max_suppressed_image(), from_host.get_width(), from_host.get_height(), OUTPUT_NON_MAX_SUPPRESSED_FILE_NAME, false);

	from_host.compute_pixel_thresholds();

	from_host.apply_double_thresholds();
#ifdef DEBUG
	fprintf(f, "double_thresholded_image\n");
	print_log_matrix(f, from_host.get_double_thresholded_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_double_thresholded_image(), from_host.get_width(), from_host.get_height(), OUTPUT_DOUBLE_THRESHOLDED_FILE_NAME, false);

	from_host.apply_hysteresis_edge_tracking();
#ifdef DEBUG
	fprintf(f, "edge_tracked_image\n");
	print_log_matrix(f, from_host.get_edge_tracked_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_edge_tracked_image(), from_host.get_width(), from_host.get_height(), OUTPUT_EDGE_TRACKED_FILE_NAME, false);

	printf("CPU took %.2fms\n", from_host.get_total_time_taken());

#ifdef DEBUG
	fclose(f);
#endif //DEBUG

	// ##################################################################################################
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

	from_device.apply_gaussian_kernel();
#ifdef DEBUG
	fprintf(f, "gaussiated_image_cuda\n");
	float *gaussiated_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(gaussiated_image_temp, from_device.get_gaussiated_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, gaussiated_image_temp, from_device.get_width(), from_device.get_height());
	free(gaussiated_image_temp);
#endif //DEBUG
	write_image_to_file(from_device.get_gaussiated_image(), from_device.get_width(), from_device.get_height(), OUTPUT_CUDA_GAUSSIATED_FILE_NAME, true);

	//from_device.compute_pixel_thresholds();

	from_device.apply_sobel_filter_x();
	//CHECK(cudaMemcpy(from_device.get_sobeled_grad_x_image(), from_host.get_sobeled_grad_x_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyHostToDevice));
#ifdef DEBUG
	fprintf(f, "sobeled_grad_x_image_cuda\n");
	float *sobeled_grad_x_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(sobeled_grad_x_image_temp, from_device.get_sobeled_grad_x_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobeled_grad_x_image_temp, from_device.get_width(), from_device.get_height());
	free(sobeled_grad_x_image_temp);
#endif //DEBUG
	write_image_to_file(from_device.get_sobeled_grad_x_image(), from_device.get_width(), from_device.get_height(), OUTPUT_CUDA_SOBELED_GRAD_X_FILE_NAME, true);

	from_device.apply_sobel_filter_y();
	//CHECK(cudaMemcpy(from_device.get_sobeled_grad_y_image(), from_host.get_sobeled_grad_y_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyHostToDevice));
#ifdef DEBUG
	fprintf(f, "sobeled_grad_y_image_cuda\n");
	float *sobeled_grad_y_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(sobeled_grad_y_image_temp, from_device.get_sobeled_grad_y_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobeled_grad_y_image_temp, from_device.get_width(), from_device.get_height());
	free(sobeled_grad_y_image_temp);
#endif //DEBUG
	write_image_to_file(from_device.get_sobeled_grad_y_image(), from_device.get_width(), from_device.get_height(), OUTPUT_CUDA_SOBELED_GRAD_Y_FILE_NAME, true);

	from_device.calculate_sobel_magnitude();
#ifdef DEBUG
	fprintf(f, "sobeled_mag_image_cuda\n");
	float *sobeled_mag_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(sobeled_mag_image_temp, from_device.get_sobeled_mag_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobeled_mag_image_temp, from_device.get_width(), from_device.get_height());
	free(sobeled_mag_image_temp);
#endif //DEBUG

	from_device.calculate_sobel_direction();
#ifdef DEBUG
	fprintf(f, "sobeled_dir_image_cuda\n");
	float *sobeled_dir_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(sobeled_dir_image_temp, from_device.get_sobeled_dir_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, sobeled_dir_image_temp, from_device.get_width(), from_device.get_height());
	free(sobeled_dir_image_temp);
#endif //DEBUG

	from_device.apply_non_max_suppression();
#ifdef DEBUG
	fprintf(f, "non_max_suppressed_image_cuda\n");
	float *non_max_suppressed_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(non_max_suppressed_image_temp, from_device.get_non_max_suppressed_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, non_max_suppressed_image_temp, from_device.get_width(), from_device.get_height());
	free(non_max_suppressed_image_temp);
#endif //DEBUG
	write_image_to_file(from_device.get_non_max_suppressed_image(), from_device.get_width(), from_device.get_height(), OUTPUT_CUDA_NON_MAX_SUPPRESSED_FILE_NAME, true);

	from_device.compute_pixel_thresholds();

	from_device.apply_double_thresholds();
#ifdef DEBUG
	fprintf(f, "double_thresholded_image_cuda\n");
	float *double_thresholded_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(double_thresholded_image_temp, from_device.get_double_thresholded_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, double_thresholded_image_temp, from_device.get_width(), from_device.get_height());
	free(double_thresholded_image_temp);
#endif //DEBUG
	write_image_to_file(from_device.get_double_thresholded_image(), from_device.get_width(), from_device.get_height(), OUTPUT_CUDA_DOUBLE_THRESHOLDED_FILE_NAME, true);

	from_device.apply_hysteresis_edge_tracking();
#ifdef DEBUG
	fprintf(f, "edge_tracked_image_cuda\n");
	float *edge_tracked_image_temp = (float*)malloc(sizeof(float) * from_device.get_width() * from_device.get_height());
	CHECK(cudaMemcpy(edge_tracked_image_temp, from_device.get_edge_tracked_image(), sizeof(float) * from_device.get_width() * from_device.get_height(), cudaMemcpyDeviceToHost));
	print_log_matrix(f, edge_tracked_image_temp, from_device.get_width(), from_device.get_height());
	free(edge_tracked_image_temp);
#endif //DEBUG
	write_image_to_file(from_device.get_edge_tracked_image(), from_device.get_width(), from_device.get_height(), OUTPUT_CUDA_EDGE_TRACKED_FILE_NAME, true);

	printf("CUDA took %.2fms\n", from_device.get_total_time_taken());

#ifdef DEBUG
	fclose(f);
#endif //DEBUG


}