#ifndef CANNYEDGE_C
#define CANNYEDGE_C

#include "BeoSub/CannyEdge.H"

#include "Util/Types.H"
#include "Util/log.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Raster/Raster.H"
#include "rutz/shared_ptr.h"
#include "BeoSub/hysteresis.H"
#include "VFAT/segmentImageTrackMC.H"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream> //needed for segmentImageTrackMC!
#include <math.h>
#include <list>
#include "Image/MatrixOps.H"

/*******************************************************************************
 * PROCEDURE: cannyEdge
 * PURPOSE: Simple wrapper for Mike's canny edge detection implementation
 * NAME: Rand Voorhies
 * DATE: 6/21/06
 *******************************************************************************/
int cannyEdge(Image<byte> &inputImage, float sigma, float tlow, float thigh, Image<byte> &outputImage) {
  unsigned char *edge;
  int ret = canny(inputImage.getArrayPtr(), inputImage.getHeight(), inputImage.getWidth(), sigma, tlow, thigh, &edge, NULL);
  outputImage = Image<unsigned char>(edge, inputImage.getWidth(), inputImage.getHeight());
  free(edge);
  return ret;
}

/*******************************************************************************
 * PROCEDURE: canny
 * PURPOSE: To perform canny edge detection.
 * NAME: Mike Heath
 * DATE: 2/15/96
 //Pradeep: returns the centroid of the "white" pixels
 *******************************************************************************/
int canny(unsigned char *image, int rows, int cols, float sigma,
          float tlow, float thigh, unsigned char **edge, char *fname)
{
  FILE *fpdir=NULL;          /* File to write the gradient image to.     */
  unsigned char *nms;        /* Points that are local maximal magnitude. */
  short int *smoothedim,     /* The image after gaussian smoothing.      */
    *delta_x,        /* The first devivative image, x-direction. */
    *delta_y,        /* The first derivative image, y-direction. */
    *magnitude;      /* The magnitude of the gadient image.      */
  float *dir_radians=NULL;   /* Gradient direction image.                */

  /****************************************************************************
   * Perform gaussian smoothing on the image using the input standard
   * deviation.
   ****************************************************************************/
  gaussian_smooth(image, rows, cols, sigma, &smoothedim);


  /****************************************************************************
   * Compute the first derivative in the x and y directions.
   ****************************************************************************/
  derrivative_x_y(smoothedim, rows, cols, &delta_x, &delta_y);




  /****************************************************************************
   * This option to write out the direction of the edge gradient was added
   * to make the information available for computing an edge quality figure
   * of merit.
   ****************************************************************************/
  if(fname != NULL){
    /*************************************************************************
     * Compute the direction up the gradient, in radians that are
     * specified counteclockwise from the positive x-axis.
     *************************************************************************/
    radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

    /*************************************************************************
     * Write the gradient direction image out to a file.
     *************************************************************************/
    if((fpdir = fopen(fname, "wb")) == NULL){
      fprintf(stderr, "Error opening the file %s for writing.\n", fname);
      exit(1);
    }
    fwrite(dir_radians, sizeof(float), rows*cols, fpdir);
    fclose(fpdir);
    free(dir_radians);
  }

  /****************************************************************************
   * Compute the magnitude of the gradient.
   ****************************************************************************/
  magnitude_x_y(delta_x, delta_y, rows, cols, &magnitude);

  /****************************************************************************
   * Perform non-maximal suppression.
   ****************************************************************************/
  if((nms = (unsigned char *) calloc(rows*cols,sizeof(unsigned char)))==NULL){
    fprintf(stderr, "Error allocating the nms image.\n");
    exit(1);
  }

  non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);

  /****************************************************************************
   * Use hysteresis to mark the edge pixels.
   ****************************************************************************/
  if((*edge=(unsigned char *)calloc(rows*cols,sizeof(unsigned char))) ==NULL){
    fprintf(stderr, "Error allocating the edge image.\n");
    exit(1);
  }
  //printf("\n%d %d %d %d %d %d\n", magnitude, nms, rows, cols, tlow, thigh);
  int centroid = apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);
  /****************************************************************************
   * Free all of the memory that we allocated except for the edge image that
   * is still being used to store out result.
   ****************************************************************************/
  free(smoothedim);
  free(delta_x);
  free(delta_y);
  free(magnitude);
  free(nms);
  return centroid;
}

/*******************************************************************************
 * Procedure: radian_direction
 * Purpose: To compute a direction of the gradient image from component dx and
 * dy images. Because not all derriviatives are computed in the same way, this
 * code allows for dx or dy to have been calculated in different ways.
 *
 * FOR X:  xdirtag = -1  for  [-1 0  1]
 *         xdirtag =  1  for  [ 1 0 -1]
 *
 * FOR Y:  ydirtag = -1  for  [-1 0  1]'
 *         ydirtag =  1  for  [ 1 0 -1]'
 *
 * The resulting angle is in radians measured counterclockwise from the
 * xdirection. The angle points "up the gradient".
 *******************************************************************************/
void radian_direction(short int *delta_x, short int *delta_y, int rows,
                      int cols, float **dir_radians, int xdirtag, int ydirtag)
{
  int r, c, pos;
  float *dirim=NULL;
  double dx, dy;

  /****************************************************************************
   * Allocate an image to store the direction of the gradient.
   ****************************************************************************/
  if((dirim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
    fprintf(stderr, "Error allocating the gradient direction image.\n");
    exit(1);
  }
  *dir_radians = dirim;

  for(r=0,pos=0;r<rows;r++){
    for(c=0;c<cols;c++,pos++){
      dx = (double)delta_x[pos];
      dy = (double)delta_y[pos];

      if(xdirtag == 1) dx = -dx;
      if(ydirtag == -1) dy = -dy;

      dirim[pos] = (float)angle_radians(dx, dy);
    }
  }
}

/*******************************************************************************
 * FUNCTION: angle_radians
 * PURPOSE: This procedure computes the angle of a vector with components x and
 * y. It returns this angle in radians with the answer being in the range
 * 0 <= angle <2*PI.
 *******************************************************************************/
double angle_radians(double x, double y)
{
  double xu, yu, ang;

  xu = fabs(x);
  yu = fabs(y);

  if((xu == 0) && (yu == 0)) return(0);

  ang = atan(yu/xu);

  if(x >= 0){
    if(y >= 0) return(ang);
    else return(2*M_PI - ang);
  }
  else{
    if(y >= 0) return(M_PI - ang);
    else return(M_PI + ang);
  }
}

/*******************************************************************************
 * PROCEDURE: magnitude_x_y
 * PURPOSE: Compute the magnitude of the gradient. This is the square root of
 * the sum of the squared derivative values.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
                   short int **magnitude)
{
  int r, c, pos, sq1, sq2;

  /****************************************************************************
   * Allocate an image to store the magnitude of the gradient.
   ****************************************************************************/
  if((*magnitude = (short *) calloc(rows*cols, sizeof(short))) == NULL){
    fprintf(stderr, "Error allocating the magnitude image.\n");
    exit(1);
  }

  for(r=0,pos=0;r<rows;r++){
    for(c=0;c<cols;c++,pos++){
      sq1 = (int)delta_x[pos] * (int)delta_x[pos];
      sq2 = (int)delta_y[pos] * (int)delta_y[pos];
      (*magnitude)[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
    }
  }

}

/*******************************************************************************
 * PROCEDURE: derrivative_x_y
 * PURPOSE: Compute the first derivative of the image in both the x any y
 * directions. The differential filters that are used are:
 *
 *                                          -1
 *         dx =  -1 0 +1     and       dy =  0
 *                                          +1
 *
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void derrivative_x_y(short int *smoothedim, int rows, int cols,
                     short int **delta_x, short int **delta_y)
{
  int r, c, pos;

  /****************************************************************************
   * Allocate images to store the derivatives.
   ****************************************************************************/
  if(((*delta_x) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
    fprintf(stderr, "Error allocating the delta_x image.\n");
    exit(1);
  }
  if(((*delta_y) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
    fprintf(stderr, "Error allocating the delta_x image.\n");
    exit(1);
  }

  /****************************************************************************
   * Compute the x-derivative. Adjust the derivative at the borders to avoid
   * losing pixels.
   ****************************************************************************/
  for(r=0;r<rows;r++){
    pos = r * cols;
    (*delta_x)[pos] = smoothedim[pos+1] - smoothedim[pos];
    pos++;
    for(c=1;c<(cols-1);c++,pos++){
      (*delta_x)[pos] = smoothedim[pos+1] - smoothedim[pos-1];
    }
    (*delta_x)[pos] = smoothedim[pos] - smoothedim[pos-1];
  }

  /****************************************************************************
   * Compute the y-derivative. Adjust the derivative at the borders to avoid
   * losing pixels.
   ****************************************************************************/
  for(c=0;c<cols;c++){
    pos = c;
    (*delta_y)[pos] = smoothedim[pos+cols] - smoothedim[pos];
    pos += cols;
    for(r=1;r<(rows-1);r++,pos+=cols){
      (*delta_y)[pos] = smoothedim[pos+cols] - smoothedim[pos-cols];
    }
    (*delta_y)[pos] = smoothedim[pos] - smoothedim[pos-cols];
  }
}

/*******************************************************************************
 * PROCEDURE: gaussian_smooth
 * PURPOSE: Blur an image with a gaussian filter.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
                     short int **smoothedim)
{
  int r, c, rr, cc,     /* Counter variables. */
    windowsize,        /* Dimension of the gaussian kernel. */
    center;            /* Half of the windowsize. */
  float *tempim,        /* Buffer for separable filter gaussian smoothing. */
    *kernel,        /* A one dimensional gaussian kernel. */
    dot,            /* Dot product summing variable. */
    sum;            /* Sum of the kernel weights variable. */

  /****************************************************************************
   * Create a 1-dimensional gaussian smoothing kernel.
   ****************************************************************************/
  make_gaussian_kernel(sigma, &kernel, &windowsize);
  center = windowsize / 2;

  /****************************************************************************
   * Allocate a temporary buffer image and the smoothed image.
   ****************************************************************************/
  if((tempim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
    fprintf(stderr, "Error allocating the buffer image.\n");
    exit(1);
  }
  if(((*smoothedim) = (short int *) calloc(rows*cols,
                                           sizeof(short int))) == NULL){
    fprintf(stderr, "Error allocating the smoothed image.\n");
    exit(1);
  }

  /****************************************************************************
   * Blur in the x - direction.
   ****************************************************************************/
  for(r=0;r<rows;r++){
    for(c=0;c<cols;c++){
      dot = 0.0;
      sum = 0.0;
      for(cc=(-center);cc<=center;cc++){
        if(((c+cc) >= 0) && ((c+cc) < cols)){
          dot += (float)image[r*cols+(c+cc)] * kernel[center+cc];
          sum += kernel[center+cc];
        }
      }
      tempim[r*cols+c] = dot/sum;
    }
  }

  /****************************************************************************
   * Blur in the y - direction.
   ****************************************************************************/
  for(c=0;c<cols;c++){
    for(r=0;r<rows;r++){
      sum = 0.0;
      dot = 0.0;
      for(rr=(-center);rr<=center;rr++){
        if(((r+rr) >= 0) && ((r+rr) < rows)){
          dot += tempim[(r+rr)*cols+c] * kernel[center+rr];
          sum += kernel[center+rr];
        }
      }
      (*smoothedim)[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
    }
  }

  free(tempim);
  free(kernel);
}

/*******************************************************************************
 * PROCEDURE: make_gaussian_kernel
 * PURPOSE: Create a one dimensional gaussian kernel.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
  int i, center;
  float x, fx, sum=0.0;

  *windowsize = int(1 + 2 * ceil(2.5 * sigma));
  center = (*windowsize) / 2;

  if((*kernel = (float *) calloc((*windowsize), sizeof(float))) == NULL){
    fprintf(stderr, "Error callocing the gaussian kernel array.\n");
    exit(1);
  }

  for(i=0;i<(*windowsize);i++){
    x = (float)(i - center);
    fx = pow(2.71828, -0.5*x*x/(sigma*sigma)) / (sigma * sqrt(6.2831853));
    (*kernel)[i] = fx;
    sum += fx;
  }

  for(i=0;i<(*windowsize);i++) (*kernel)[i] /= sum;

}


#endif
