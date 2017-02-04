/**
 * @file    main.cpp
 * @author  Everton Nagel, Gerson Beckenkamp and Guilherme Ritter
 * @brief   Laser cast implementation
 */

#include <stdlib.h>
#include <stdio.h>

#include <unistd.h>
#include <fcntl.h>

/// Frame buffer library.
#include <linux/fb.h>

#include <sys/mman.h>
#include <sys/ioctl.h>

/// Open computer vision (CV) library.
#include "opencv2/opencv.hpp"

#define RES_X 800                       ///< Screen horizontal resolution.
#define RES_Y 600                       ///< Screen vertical resolution.
#define CATETO_CLEAN 64                 ///< Square size of clean command.
#define CATETO_CLOSE (CATETO_CLEAN / 2) ///< Square size of close command.
#define CAMERA_RES_X 640                ///< Camera horizontal resolution.
#define CAMERA_RES_Y 480                ///< Camera vertical resolution.

using namespace cv;
using namespace std;

typedef struct {
	int x_ini, x_fim, y_ini, y_fim;
} Retangulo;

typedef struct {
	int x, y;
} Ponto;

/**
 * @brief   Get a red point from a camera frame.
 * @param   frame   [IN]    OpenCV generic Mat object.
 * @param   point   [OUT]   A dot pointer to fill.
 * @param   out     [OUT]   The vector containing all traced dots.
 */
int getRed(Mat frame, Ponto *point, std::vector<Ponto> &out)
{
    // walk the matrix until find a red fragment
	for (int y = 0; y < frame.rows; y++){
        for (int x = 0; x < frame.cols; x++) {

        	if(frame.at<Vec3b>(y,x)[2]>200){

                if(y <= CATETO_CLEAN && x <= CATETO_CLEAN){
                    // clean command
                    out.clear(); // clean all traced dots
                    system("xrefresh"); // force a screen update to remove laser prints
                }
                else if(y <= CATETO_CLOSE && x > CAMERA_RES_X - CATETO_CLOSE){
                    return -1; // close command
                }
                else {
                    // set the new laser dot
                    point->x = x;
                    point->y = y;
                }
        		return 1;
        	}
       	}
    }
    // no laser trace was found
    return 0;
}

/**
 * @brief   Add rectangle valid points and get the vector to print.
 * @param   frame   [IN]    OpenCV generic Mat as a frame.
 * @param   out     [OUT]   The vector containing all traced dots.
 * @param   rect    [IN]    The rectangle with the laser points.
 */
void getVet(Mat frame, std::vector<Ponto> &out, Retangulo *rect)
{
	for (int y = rect->y_ini; y < rect->y_fim; y++) {
        for (int x = rect->x_ini; x < rect->x_fim; x++) {
        	if(frame.at<Vec3b>(y,x)[2]>200){
                Ponto p;

                // wrapper Camera <----> Screen
                p.x = roundf(x * RES_X / CAMERA_RES_X);
                p.y = roundf(y * RES_Y / CAMERA_RES_Y);
                out.push_back(p); // add the new pixel
        	}
       	}
    }
}

/**
 * @brief   Get last red point.
 * @param   frame   [IN]    OpenCV generic Mat object.
 * @param   point   [IN]    A dot pointer to fill.
 * @param   up_x    [IN]    Horizontal position.
 * @param   up_y    [IN]    Vertical position.
 * @param   out     [OUT]   The vector containing all traced dots.
 */
void getLastRed(Mat frame, Ponto *point, int up_x, int up_y, Retangulo *out)
{
	int max_x, min_x, max_y,min_y;
	int x_atual = point->x;
    int y_atual = point->y;

	max_x = min_x = point->x;
	max_y = min_y = point->y;

	while(frame.at<Vec3b>(y_atual,x_atual)[2]>200){
		if(max_x<x_atual)
			max_x = x_atual;
		if(min_x>x_atual)
			min_x = x_atual;
		if(max_y<y_atual)
			max_y = y_atual;
		if(min_y>y_atual)
			min_y = y_atual;

        if (x_atual + up_x >= 0 && x_atual + up_x < frame.cols) {
            x_atual+=up_x;
        }
        else {
            break;
        }
        if (y_atual + up_y >= 0 && y_atual + up_y < frame.rows) {
            y_atual+=up_y;
        }
        else {
            break;
        }
	}

    // care to not overflow
	if(out->x_ini>min_x)
		out->x_ini = min_x;
	if(out->x_fim<max_x)
		out->x_fim = max_x;
	if(out->y_ini>min_y)
		out->y_ini = min_y;
	if(out->y_fim<max_y)
		out->y_fim = max_y;
}

/**
 * @brief   Get the other points around a raw point.
 * @param   frame   [IN]    OpenCV generic Mat object.
 * @param   point   [IN]    The captured point.
 * @param   out     [OUT]   The rectangle points.
 */
void getRect(Mat frame, Ponto *point, Retangulo *rect)
{
	rect->x_ini = point->x;
	rect->x_fim = point->x;
	rect->y_ini = point->y;
	rect->y_fim = point->y;

	getLastRed(frame, point, 2, 0, rect);
	getLastRed(frame, point, -2, 0, rect);
	getLastRed(frame, point, 0, 2, rect);
	getLastRed(frame, point, 0, -2, rect);
	getLastRed(frame, point, 2, 2, rect);
	getLastRed(frame, point, -2, 2, rect);
	getLastRed(frame, point, 2, -2, rect);
	getLastRed(frame, point, -2, -2, rect);

    // care to not overflow the screen limits
	if(rect->x_ini-3>=0)
		rect->x_ini-=3;
	if(rect->y_ini-3>=0)
		rect->y_ini-=3;
	if(rect->x_fim+3<frame.cols)
		rect->x_fim+=3;
	if(rect->y_fim+3<frame.rows)
		rect->y_fim+=3;
}

int main (int argc, char ** argv)
{
	VideoCapture cap(0); // webcam capture
	Mat frame;
    std::vector<Ponto> framePoints;
	int fbfd = 0;
    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;
    long int screensize = 0;
    char *fbp = 0;
    long int location = 0;
    int rc;

	if (!cap.isOpened()) {
        std::cout << "failed to open file: " << argv[1] << std::endl;
        return -1;
    }

    // open frame buffer for reading and writing
    fbfd = open("/dev/fb0", O_RDWR);
    if (fbfd == -1) {
        perror("Error: cannot open framebuffer device");
        exit(1);
    }
    printf("The framebuffer device was opened successfully.\n");

    // Get fixed screen information
    if (ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo) == -1) {
        perror("Error reading fixed information");
        exit(2);
    }

    // Get variable screen information
    if (ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo) == -1) {
        perror("Error reading variable information");
        exit(3);
    }
    Ponto point;
    Retangulo rect;

    // Figure out the size of the screen in bytes
    screensize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8;

    for (;;) { // start the continuous loop routine to track the laser
		cap >> frame;
		if((rc = getRed(frame, &point, framePoints)) == 1) {
            // a red point was found!
			getRect(frame, &point, &rect); // get a rectangle from the point
			getVet(frame, framePoints, &rect); // add the new points to update
        }
        else if (rc == -1) {
            // clean command
            framePoints.clear();
            system("xrefresh");
            break;
        }

        /**
         * OBS: We should update screen even if no laser dot was found for
         * the current camera frame. The previous vector points will be
         * printed.
         */

        // Map the device to memory
        fbp = (char *)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fbfd, 0);
        for (int idx = 0; idx < framePoints.size(); idx++) {

            // new laser dot location
            location = (framePoints[idx].x+vinfo.xoffset) * (vinfo.bits_per_pixel/8) +
                (framePoints[idx].y+vinfo.yoffset) * finfo.line_length;

            if (vinfo.bits_per_pixel == 32) {
                // do it cyan
                *(fbp + location)     = 255; // blue
                *(fbp + location + 1) = 255; // green
                *(fbp + location + 2) = 0;   // red
                *(fbp + location + 3) = 0;   // no transparency
                location += 4;
            } else {
                // assume as 16 bits per pixel (bpp)
                int b = 255;
                int g = 255;
                int r = 0;
                unsigned short int t = r<<11 | g << 5 | b;
                *((unsigned short int*)(fbp + location)) = t;
            }
        }
        munmap(fbp, screensize);
    }
    close(fbfd);
    return 0;
}
