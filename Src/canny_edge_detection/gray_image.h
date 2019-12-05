/* creates a basic handler for a grayscale image
*/

#ifndef __GRAY_IMAGE_H
#define __GRAY_IMAGE_H

class gray_image {
	private:
		float *h_gimage;
		int width;
		int height;

	public:
		gray_image(float *h_gimage, int width, int height);
		~gray_image();

		float *get_host_gimage();
		int get_width();
		int get_height();
		int get_pixel(int x, int y);
};

#endif //__GRAY_IMAGE_H