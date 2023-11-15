/*!@file BeoSub/test-hough.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-hough.C $
// $Id: test-hough.C 14376 2011-01-11 02:44:34Z pez $
//

#ifndef BEOSUB_TEST_HOUGH_C_DEFINED
#define BEOSUB_TEST_HOUGH_C_DEFINED

#include "GUI/XWindow.H"
//CAMERA STUFF
#include "Image/Image.H"
#include "Image/Pixels.H"

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberFactory.H"
#include "Util/Timer.H"
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

#include "MBARI/Geometry2D.H"
#include "rutz/compat_cmath.h" // for isnan()

using std::cout;
using std::endl;
using std::list;

//END CAMERA STUFF
//canny
#define BOOSTBLURFACTOR 90.0
#define FREE_ARG char*
#define PI 3.14

XWindow *xwin;
XWindow *xwin2;
XWindow *xwin3;

void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int **smoothedim);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols,
                     short int **delta_x, short int **delta_y);
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
                   short int **magnitude);
void radian_direction(short int *delta_x, short int *delta_y, int rows,
                      int cols, float **dir_radians, int xdirtag, int ydirtag);
void radian_direction(short int *delta_x, short int *delta_y, int rows,
                      int cols, float **dir_radians, int xdirtag, int ydirtag);
double angle_radians(double x, double y);

std::vector <LineSegment2D> houghTransform(Image<byte> &inputImage, float thetaRes, float dRes, int threshold, Image< PixRGB<byte> > &output) ;

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



//finds the line by using hough transform
//thetares is the resolution of each theta
//dRes is the resolution of the D
//returns the number of lines found
std::vector <LineSegment2D> houghTransform(Image<byte> &inputImage, float thetaRes, float dRes, int threshold, Image< PixRGB<byte> > &output)
{
        int r;
        //get the total number of angles and D from the resolution of theta and D's
        int numangle = (int) (PI / thetaRes); //in radians
        int numD = (int) (((inputImage.getWidth() + inputImage.getHeight()) * 2 + 1) / dRes);

        std::vector <LineSegment2D> lines;
   cout << "Performing Hough Line Transform:" << endl;
   cout << "numD: " << numD << endl;
   cout << "numangle: " << numangle << endl;

        //accumulator -> represents the hough space
        int accumulator[numD][numangle];

        for(int i = 0; i<numD; i++)
          for(int j = 0; j < numangle; j++)
            accumulator[i][j] = 0;

        //equation of the line is Xcos(theta)+ysin(theta) = D
        // fill the accumulator
        //        printf("numD is %d\n",numD);
    for(int j = 0; j < inputImage.getHeight(); j++ )
        for(int i = 0; i < inputImage.getWidth(); i++ )
        {
            if( inputImage.getVal(i,j) != 0 )//point is an edge
                for(int n = 0; n < numangle; n++ )//get all possible D and theta that can fit to that point
                {
                    r =  (int)(i * cos(n*thetaRes) + j * sin(n*thetaRes));
                    r += (numD -1)/2;
                    accumulator[r][n]++;
                }
        }

    Point2D<int> p1;
    Point2D<int> p2;
    Point2D<int> storage[inputImage.getWidth()*inputImage.getHeight()];
    double a;
    int totalLines = 0;
    int pointCount = 0;

    // find the peaks, ie any number greater than the threshold is considered a line
    for(int i = 0; i<numD; i++)
      for(int j = 0; j < numangle; j++)
        {
          if(accumulator[i][j] > threshold && accumulator[i][j]> accumulator[i-1][j]
             && accumulator[i][j]> accumulator[i+1][j] && accumulator[i][j]> accumulator[i][j-1]
             && accumulator[i][j]> accumulator[i][j+1])//found a peak
            {
              totalLines++;

              //get the image coordinates for the 2 endpoint of the line
              a = cos(j*thetaRes);
              // b = sin(j*thetaRes);

              for(int x = 0; x < inputImage.getWidth(); x++)
                for(int y = 0; y < inputImage.getHeight(); y++)
                  if( inputImage.getVal(x,y) != 0 )//point is an edge
                    if(((int)(x * cos(j*thetaRes) + y * sin(j*thetaRes))) + ((numD -1)/2) == i)
                      {
                        storage[pointCount].i = x;
                        storage[pointCount].j = y;
                        pointCount++;
                      }

              //check the distance between each point
              bool isolated = true;
              int numtoDiscard = 0;
              float discardPoints[inputImage.getWidth() * inputImage.getHeight()];
              for(int y = 0; y < pointCount; y++){
                for(int x = 0; x <pointCount; x++){
                  if(storage[0].distance(storage[x]) < 1 && x != y)
                    isolated = false;

                }
                if(isolated)//discard this point
                  {
                    discardPoints[numtoDiscard++] = y;
                  }
              }



              //make sure that the point is not in the to be discarded list
              for(int x=0; x < numtoDiscard; x++)
                for(int y = 0; y < pointCount; y++)
                  if(discardPoints[x] != y){
                    p1.i = p2.i = storage[y].i;
                    p1.j = p2.j = storage[y].j;

                  }
              //find the 2 endpoints of line
              //check if vertical
              if(fabs(a) < .001)
                {

                  //search the 2 endpoints
                  for(int x = 0; x < pointCount; x++)
                    {
                      bool discard = false;
                      for(int k = 0; k < numtoDiscard; k++)
                        if(discardPoints[k] == x)
                          discard = true;
                      if(!discard){
                        if(p1.j < storage[x].j)
                          {
                            p1.j = storage[x].j;
                            p1.i = storage[x].i;
                          }
                        if(p2.j > storage[x].j)
                          {
                          p2.j = storage[x].j;
                          p2.i = storage[x].i;

                          }
                      }
                    }

                }
              else // horizontal
                {
                  //search the 2 endpoints
                  for(int x = 0; x < pointCount; x++)
                    {
                      bool discard = false;
                      for(int k = 0; k < numtoDiscard; k++)
                        if(discardPoints[k] == x)
                          discard = true;
                      if(!discard){

                        if(p1.i < storage[x].i)
                          {
                            p1.j = storage[x].j;
                            p1.i = storage[x].i;
                          }
                        if(p2.i > storage[x].i)
                          {
                            p2.j = storage[x].j;
                            p2.i = storage[x].i;

                          }
                      }
                    }
                }
              pointCount = 0;
              LineSegment2D thisLine(p1, p2);
              lines.push_back(thisLine);
             drawLine(output,p1, p2,  PixRGB<byte> (255,0,0), 1);


            }
        }
     return lines;

}

/*
functions to find the best fit line
*/

double sigma(double* r, int N) {
   double sum=0.0;
   for(int i=0;i<N;++i)
    sum += r[i];
   return(sum);
}
double sigma(double* r,double* s, int N) {
   double sum=0.0;
   for(int i=0;i<N;++i)
    sum += r[i]*s[i];
   return(sum);
}
void findTanLine(int N, double *x, double *y, float &slope, float &Intercept) {
   float delta=N*sigma(x,x, N)-pow(sigma(x,N),2.0);
   Intercept =  (1.0/delta)*(sigma(x,x,N)*sigma(y,N)-sigma(x,N)*sigma(x,y,N));
   slope =  (1.0/delta)*(N*sigma(x,y, N)-sigma(x, N)*sigma(y,N));
}


/* note that the utility functions below are taken from Robert A. Mclaughlin's code on RHT.
Only slight modification is done to them
*/


/* Find largest (absolute) entry in a 2x2 matrix.
 * Return co-ords of entry in (i,j) (row, column)
   -Robert A. McLaughlin
 */
void
find_biggest_entry_2x2(double m[][3], int *i, int *j)
{
    if ((fabs(m[1][1]) > fabs(m[1][2])) && (fabs(m[1][1]) > fabs(m[2][1])) && (fabs(m[1][1]) > fabs(m[2][2])))
        {
        *i = 1;
        *j = 1;
        }
    else if ((fabs(m[1][2]) > fabs(m[2][1])) && (fabs(m[1][2]) > fabs(m[2][2])))
        {
        *i = 1;
        *j = 2;
        }
    else if (fabs(m[2][1]) > fabs(m[2][2]))
        {
        *i = 2;
        *j = 1;
        }
    else
        {
        *i = 2;
        *j = 2;
        }

}   /* end of find_biggest_entry_2x2() */

/*
 function that finds the eigenvectors
 -Robert A. McLaughlin
*/
void
find_eigenvectors_2x2_positive_semi_def_real_matrix(double **covar, double *eig_val, double **eig_vec)
{
    double  a, b, c, d;
    int     i, j;
    double  tmp[3][3];

    /* Find eigenvalues
     * Use characteristic equation.
     */
    a = 1;
    b = -covar[1][1] - covar[2][2];
    c = covar[1][1]*covar[2][2] - covar[1][2]*covar[2][1];

    d = squareOf(b) - 4*a*c;
    if (d <= 10e-5)
        {
        eig_val[1] = -b / 2.0;
        eig_val[2] = -b / 2.0;
        eig_vec[1][1] = 1.0;  eig_vec[1][2] = 0.0;
        eig_vec[2][1] = 0.0;  eig_vec[2][2] = 1.0;
        }
    else
        {
        eig_val[1] = (-b + sqrt(d)) / 2.0;
        eig_val[2] = (-b - sqrt(d)) / 2.0;

        /* Put eigenvalues in descending order.
         */
        if (eig_val[1] < eig_val[2])
            {
            d = eig_val[1];
            eig_val[1] = eig_val[2];
            eig_val[2] = d;
            }

        /* Find first eigenvector.
         */
        tmp[1][1] = covar[1][1] - eig_val[1];
        tmp[1][2] = covar[1][2];
        tmp[2][1] = covar[2][1];
        tmp[2][2] = covar[2][2] - eig_val[1];


        find_biggest_entry_2x2(tmp, &i, &j);
        if (j == 1)
            {
            eig_vec[2][1] = 1.0;
            eig_vec[1][1] = -tmp[i][2] / tmp[i][1];
            }
        else    /* j == 2 */
            {
            eig_vec[1][1] = 1.0;
            eig_vec[2][1] = -tmp[i][1] / tmp[i][2];
            }
        /* Normalise eigenvecotr.
         */
        d = sqrt(squareOf(eig_vec[1][1]) + squareOf(eig_vec[2][1]));
        eig_vec[1][1] /= d;
        eig_vec[2][1] /= d;


        /* Find secind eigenvector.
         */
        eig_vec[1][2] = -eig_vec[2][1];
        eig_vec[2][2] = eig_vec[1][1];

        }



}   /* end of find_eigenvectors_2x2_positive_semi_def_real_matrix() */


void
SwapRows(double *A, double *B, int els)
{
    double temp;
    int a;

    for(a=1;a<=els;a++)
        {
        temp=A[a];
        A[a]=B[a];
        B[a]=temp;
        }
}   /* end of SwapRows() */


void
TakeWeightRow(double *A, double *B, double num, double denom, int els)
{
    int a;

    for(a=1;a<=els;a++)
        A[a]=(A[a]*denom-B[a]*num)/denom;

}   /* end of TakeWeightRow() */

void
ScaleRow(double *A, double fact, int els)
{
    int a;

    for(a=1;a<=els;a++)
        A[a]/=fact;

}   /* end of ScaleRow() */
double **
alloc_array(int row, int column)
{
    double  **rowp;
    int     a;


    if ( (rowp=(double **)malloc((unsigned )(row+1) * sizeof(double *)) ) == NULL)
        return(NULL);
    for(a=1;a<=row;a++)
        {
        if ((rowp[a]=(double *)malloc((unsigned)(column+1)*sizeof(double))) == NULL)
            return(NULL);
        }

    return(rowp);

}   /* end of alloc_array() */


void
free_array(double **A, int row)
{
    int i;

    for (i=1; i <= row; i++)
        free(A[i]);
    free(A);

}       /* end of free_array() */

double  *alloc_vector(int dim)
{
    return( (double *)malloc(( (unsigned )(dim+1) ) * sizeof(double)) );
}

/*
  finds the inverse of matrix taken from -Robert A. McLaughlin's code
*/
void
find_inverse(double **MAT, double **I, int dim)
{

    int     a,b,R;
    double  num,denom;
    double  **M;

    M = alloc_array(dim, dim);
    for (a=1; a<=dim; a++)
        for (b=1; b<=dim; b++)
           M[a][b] = MAT[a][b];

    for(R=1;R<=dim;R++)
        for(a=1;a<=dim;a++)
            I[R][a]=(R==a)?1:0;
    for(R=1;R<=dim;R++)
        {
        if(M[R][R]==0)
            {
            for(a=R+1;a<=dim;a++)
                if (M[a][R]!=0) break;
            if (a==(dim+1))
                {
                fprintf(stderr, "Matrix non-invertable, fail on row %d left.\n",R);
                fprintf(stderr, "in find_inverse() in lin_algebra.c\n");
                exit(1);
                }
            SwapRows(M[R],M[a],dim);
            SwapRows(I[R],I[a],dim);
            }
        denom=M[R][R];
        for(a=R+1;a<=dim;a++)
            {
            num=M[a][R];
            TakeWeightRow(M[a],M[R],num,denom,dim);
            TakeWeightRow(I[a],I[R],num,denom,dim);
            }
        }
    for(R=dim;R>=1;R--)
        {
        for(a=R+1;a<=dim;a++)
            {
            num=M[R][a];
            denom=M[a][a];
            TakeWeightRow(M[R],M[a],num,denom,dim);
            TakeWeightRow(I[R],I[a],num,denom,dim);
            }
        }
    for(R=1;R<=dim;R++)
        ScaleRow(I[R],M[R][R],dim);

    free_array(M, dim);


}   /* end of find_inverse() */



/*
function that does LU decomposition
taken from Robert A. McLaughlin's code
*/
#define TINY_VALUE 1.0e-20;
int
lu_decomposition(double **M, int n, int *indx, double *d)
{
    int     i, imax, j, k;
    double  big,dum,sum,temp;
    double  *vec;

    vec = alloc_vector(n);
    *d = 1.0;
    for (i=1; i <= n; i++)
        {
        big=0.0;
        for (j=1; j <= n; j++)
            if ((temp=fabs(M[i][j])) > big) big=temp;
        if (big == 0.0)
            {
            /* Singular matrix in routine lu_decomposition
             */
            return(-1);
            }
        vec[i]=1.0/big;
        }
    for (j=1; j <= n; j++)
        {
        for (i=1; i < j; i++)
            {
            sum=M[i][j];
            for (k=1;k<i;k++)
                sum -= M[i][k]*M[k][j];
            M[i][j]=sum;
            }
        big=0.0;
        imax = 0;
        for (i=j; i <= n; i++)
            {
            sum = M[i][j];
            for (k=1; k < j; k++)
                sum -= M[i][k]*M[k][j];
            M[i][j]=sum;
            if ( (dum=vec[i]*fabs(sum)) >= big)
                {
                big = dum;
                imax = i;
                }
            }
        if (j != imax)
            {
            for (k=1; k <= n; k++)
                {
                dum=M[imax][k];
                M[imax][k]=M[j][k];
                M[j][k]=dum;
                }
            *d = -(*d);
            vec[imax]=vec[j];
            }
        indx[j]=imax;
        if (M[j][j] == 0.0)
            M[j][j] = TINY_VALUE;
        if (j != n)
            {
            dum=1.0/(M[j][j]);
            for (i=j+1; i<=n; i++)
                M[i][j] *= dum;
            }
        }
    free(vec);


    return(0);

}           /* end of lu_decomposition() */
#undef TINY_VALUE


/*
function that does LU back substitution
taken from Robert A. McLaughlin's code
*/
void
lu_back_substitution(double **M, int n, int *indx, double *b)
{
    int     i, ii=0, ip, j;
    double  sum;

    for (i=1;i<=n;i++)
        {
        ip=indx[i];
        sum=b[ip];
        b[ip]=b[i];
        if (ii)
            for (j=ii;j<=i-1;j++)
                sum -= M[i][j]*b[j];
        else if (sum) ii=i;
        b[i] = sum;
        }
    for (i=n; i >=1 ;i--)
        {
        sum = b[i];
        for (j=i+1;j<=n;j++)
            sum -= M[i][j]*b[j];
        b[i] = sum/M[i][i];
        }
}           /* end of lu_decomposition() */

/* function that transforms the a,b,c parameter of the ellipse equation to r1,r2,theta
taken from Robert A. McLaughlin's code
*/
bool
transform_ellipse_parameters(double a, double b, double c,
                                    double *r1, double *r2, double *theta)
{
    double  **M, **C, **eig_vec;
    double  *eig_val;


    M = alloc_array(2, 2);
    C = alloc_array(2, 2);
    eig_vec = alloc_array(2, 2);
    eig_val = alloc_vector(2);

    if ((M == NULL) || (C == NULL) || (eig_val == NULL) || (eig_vec == NULL))
        {
        fprintf(stderr, "malloc failed in transform_ellipse_parameters() in rht.c\n");
        exit(1);
        }


    M[1][1] = a;
    M[1][2] = M[2][1] = b;
    M[2][2] = c;
    find_inverse(M, C, 2);
    find_eigenvectors_2x2_positive_semi_def_real_matrix(C, eig_val, eig_vec);

    if ((eig_val[1] <= 0) || (eig_val[2] <= 0))
        {
        return( false );
        }


    *r1 = sqrt(eig_val[1]);
    *r2 = sqrt(eig_val[2]);
    *theta = atan2(eig_vec[2][1], eig_vec[1][1]);


    free_array(M, 2);
    free_array(C, 2);
    free_array(eig_vec, 2);
    free(eig_val);


    return( true );

}       /* end of transform_ellipse_parameters() */






//ellipse parameter node
struct ellipse {
  int x, y, confidence;
  double r1, r2, theta;
};


void drawEllipse(Image< PixRGB<byte> > &output, list<ellipse> &houghSpace)
{

  list<ellipse>::iterator Iter;
  //draw the ellipses
  for(Iter = houghSpace.begin(); Iter != houghSpace.end(); Iter++)
    {
      //cout<<"drawing "<<endl;
      //drawCircle(output,Point2D<int>((*Iter).x,(*Iter).y),(int)(*Iter).c,PixRGB<byte> (255,0,0));
      // for(int u = 0; u < inputImage.getWidth(); u++)
      //for(int v = 0; v < inputImage.getHeight(); v++)
      //  if(((int)((*Iter).a * pow((float)(u - (*Iter).x),2)) + (2 * (*Iter).b * (u - (*Iter).x) * (v- (*Iter).y)) + ((*Iter).c * pow((float)(v-(*Iter).y),2)))== 1 && (*Iter).confidence > threshold){
      //    drawPoint(output,u,v,PixRGB<byte> (255,0,0));
            //   cout<<"drawing confidence is "<< (*Iter).confidence<<endl;
      //  }

      int width = 20;
      int                 x, y;
      int                 x1, y1, x2, y2;
      double              perimeter, incr, grad, rho;
      double              M[3][3];
      int                 offset;
      bool error = false;

      if (((*Iter).r1 <= 20) || ((*Iter).r2 <= 20))
        error = true;


      /* The ellipse will be contructed as a sequence of
       * line segments.
       * incr indicates how much of the ellipse
       * each line segments will cover.
       */
      perimeter = 3.14159 * ( 3*(fabs((*Iter).r1)+fabs((*Iter).r2)) - sqrt( (fabs((*Iter).r1)+3*fabs((*Iter).r2))*(3*fabs((*Iter).r1)+fabs((*Iter).r2)) ) );
      // cout<<"p is "<<(*Iter).x<<"q is "<<(*Iter).y<<"r1 is "<<(*Iter).r1<<"r2 is "<<(*Iter).r2<<"theta is "<<(*Iter).theta<<endl;
      // cout<<"confidence is "<<(*Iter).confidence<<endl;
      /* Error check.
       */
      //cout<<"perimeter is "<<perimeter<<endl;
      if (perimeter <= 20)
        error = true;


      if(!error && (*Iter).confidence > 10)
        {
          //          cout<<"!error"<<endl;
          incr = 60.0/perimeter;


          /* The ellipse is defined as all points [x,y] given by:
           *
           * { [x,y] : for all theta in [0, 2pi),
           *           [x,y] = E . D . E^-1 . [cos(theta), sin (theta)}
           *
           * where E is the matrix containing the direction of
           * the pricipal axes
           * of the ellipse
           * and D is:   -    -
           *            |r1  0 |
           *            | 0  r2|
           *             -    -
           * First calculate E . D . E^-1
           */
          M[1][1] = (*Iter).r1*pow(cos((*Iter).theta),2) + (*Iter).r2*pow(sin((*Iter).theta),2);
          M[1][2] = M[2][1] = ((*Iter).r1-(*Iter).r2)*sin((*Iter).theta)*cos((*Iter).theta);
          M[2][2] = (*Iter).r1*(pow(sin((*Iter).theta),2)) + (*Iter).r2*(pow(cos((*Iter).theta),2));

          x1 = (int )((*Iter).x + M[1][1]*cos(0.0) + M[1][2]*sin(0.0));
          y1 = (int )((*Iter).y + M[2][1]*cos(0.0) + M[2][2]*sin(0.0));


          for (rho=incr; rho < (2*M_PI + incr); rho += incr)
            {
              x2 = (int )((*Iter).x + M[1][1]*cos(rho) + M[1][2]*sin(rho));
              y2 = (int )((*Iter).y + M[2][1]*cos(rho) + M[2][2]*sin(rho));
              if ((x1 >= 0) && (x1 < output.getWidth()) && (y1 >= 0) && (y1 < output.getHeight()) &&
                  (x2 >= 0) && (x2 < output.getWidth()) && (y2 >= 0) && (y2 < output.getHeight()))
                {
                  if (fabs(x2-x1) >= fabs(y2-y1))
                    {
                      /* x changes more than y
                       */
                      if (x1 < x2)
                        {
                          grad = ((double )(y2-y1)) / (double )(x2-x1);
                          for (x=x1; x <= x2; x++)
                            {
                              y = y1 + (int )(grad*(double )(x - x1));
                              for (offset=-width/2; offset < (width+1)/2; offset++)
                                {
                                  if (((y+offset) >= 0) && ((y+offset) < output.getHeight()))
                                    {
                                      drawPoint(output,x,y/*+offset*/,PixRGB<byte> (255,0,0));
                                      //                                      cout<<"DRAWING POINT"<<endl;
                                    }
                                }
                            }
                        }
                      else if (x1 > x2)
                        {
                          grad = ((double )(y1-y2)) / (double )(x1-x2);
                          for (x=x2; x <= x1; x++)
                            {
                              y = y2 + (int )(grad*(double )(x - x2));
                              for (offset=-width/2; offset < (width+1)/2; offset++)
                                {
                                  if (((y+offset) >= 0) && ((y+offset) < output.getHeight()))
                                    {
                                      drawPoint(output,x,y/*+offset*/,PixRGB<byte> (255,0,0));
                                      // cout<<"DRAWING POINT"<<endl;
                                    }
                                }
                            }
                        }
                    }
                  else        /* if (abs(x2-x1) < abs(y2-y1)) */
                    {
                      /* y changes more than x
                       */
                      if (y1 < y2)
                        {
                          grad = ((double )(x2-x1)) / (double )(y2-y1);
                          for (y=y1; y <= y2; y++)
                            {
                              x = x1 + (int )(grad*(double )(y-y1));
                              for (offset=-width/2; offset < (width+1)/2; offset++)
                                {
                                  if (((x+offset) >= 0) && ((x+offset) < output.getWidth()))
                                    {
                                      drawPoint(output,x/*+offset*/,y,PixRGB<byte> (255,0,0));
                                      //cout<<"DRAWING POINT"<<endl;
                                    }
                                }
                            }
                        }
                      else if (y1 > y2)
                        {
                          grad = ((double )(x1-x2)) / (double )(y1-y2);
                          for (y=y2; y <= y1; y++)
                            {
                              x = x2 + (int )(grad*(double )(y - y2));
                              for (offset=-width/2; offset < (width+1)/2; offset++)
                                {
                                  if (((x+offset) >= 0) && ((x+offset) < output.getWidth()))
                                    {
                                      drawPoint(output,x/*+offset*/,y,PixRGB<byte> (255,0,0));
                                      // cout<<"DRAWING POINT"<<endl;
                                    }
                                }
                            }
                        }
                    }
                }   /* end of 'if ((x1 >= 0) && (x1 < aImage->x) &&...' */
              x1 = x2;
              y1 = y2;
            }
        }
    }

}


int
count_pixels(Image<byte> &output, double x_centre, double y_centre,
                                        double r1, double r2, double theta)
{
/*
 *      x, y;       -- centre of ellipse
 *      r1, r2;     -- major/minor axis
 *      theta;      -- angle
 */
  int width = 20;
  int             x, y;
  int             x1, y1, x2, y2;
  double          perimeter, incr, grad, rho;
  double          M[3][3];
  int             offset;
  int             num_pixels = 0;

  if ((r1 <= width) || (r2 <= width))
    return(0);



    /* The ellipse will be contructed as a sequence of
     * line segments.
     * incr indicates how much of the ellipse
     * each line segments will cover.
     */
  perimeter = M_PI * ( 3*(fabs(r1)+fabs(r2)) - sqrt( (fabs(r1)+3*fabs(r2))*(3*fabs(r1)+fabs(r2)) ) );


    /* Error check.
     */
  if (perimeter <= width)
    return( 0 );        /* zero width or length ellipse.
                             * Don't bother counting pixels.
                             */



  incr = 60.0/perimeter;


    /* The ellipse is defined as all points [x,y] given by:
     *
     * { [x,y] : for all theta in [0, 2pi),
     *           [x,y] = E . D . E^-1 . [cos(theta), sin (theta)}
     *
     * where E is the matrix containing the direction of
     * the pricipal axes
     * of the ellipse
     * and D is:   -    -
     *            |r1  0 |
     *            | 0  r2|
     *             -    -
     * First calculate E . D . E^-1
     */
  M[1][1] = r1*squareOf(cos(theta)) + r2*squareOf(sin(theta));
  M[1][2] = M[2][1] = (r1-r2)*sin(theta)*cos(theta);
  M[2][2] = r1*squareOf(sin(theta)) + r2*squareOf(cos(theta));

  x1 = (int )(x_centre + M[1][1]*cos(0.0) + M[1][2]*sin(0.0));
  y1 = (int )(y_centre + M[2][1]*cos(0.0) + M[2][2]*sin(0.0));


  for (rho=incr; rho < (2*M_PI + incr); rho += incr)
    {
      x2 = (int )(x_centre + M[1][1]*cos(rho) + M[1][2]*sin(rho));
      y2 = (int )(y_centre + M[2][1]*cos(rho) + M[2][2]*sin(rho));
      if ((x1 >= 0) && (x1 < output.getWidth()) && (y1 >= 0) && (y1 < output.getHeight()) &&
          (x2 >= 0) && (x2 < output.getWidth()) && (y2 >= 0) && (y2 < output.getHeight()))
        {
          if (abs(x2-x1) >= abs(y2-y1))
            {
              /* x changes more than y
               */
              if (x1 < x2)
                {
                  grad = ((double )(y2-y1)) / (double )(x2-x1);
                  for (x=x1; x <= x2; x++)
                    {
                      y = y1 + (int )(grad*(double )(x - x1));
                      for (offset=-width/2; offset < (width+1)/2; offset++)
                        {
                          if (((y+offset) >= 0) && ((y+offset) < output.getHeight()))
                            {
                              if ((output.getVal(x,y+offset) == 255))
                                {
                                  num_pixels++;
                                }
                            }
                        }
                    }
                }
              else if (x1 > x2)
                {
                  grad = ((double )(y1-y2)) / (double )(x1-x2);
                  for (x=x2; x <= x1; x++)
                    {
                      y = y2 + (int )(grad*(double )(x - x2));
                      for (offset=-width/2; offset < (width+1)/2; offset++)
                        {
                            if (((y+offset) >= 0) && ((y+offset) < output.getHeight()))
                              {
                                if ((output.getVal(x,y+offset) == 255))
                                  {
                                    num_pixels++;
                                  }
                              }
                        }
                    }
                }
            }
          else        /* if (abs(x2-x1) < abs(y2-y1)) */
            {
              /* y changes more than x
               */
              if (y1 < y2)
                {
                  grad = ((double )(x2-x1)) / (double )(y2-y1);
                  for (y=y1; y <= y2; y++)
                    {
                      x = x1 + (int )(grad*(double )(y-y1));
                      for (offset=-width/2; offset < (width+1)/2; offset++)
                              {
                          if (((x+offset) >= 0) && ((x+offset) < output.getWidth()))
                            {
                              if ((output.getVal(x+offset,y) == 255))
                                {
                                  num_pixels++;
                                }
                            }
                        }
                    }
                }
              else if (y1 > y2)
                    {
                      grad = ((double )(x1-x2)) / (double )(y1-y2);
                      for (y=y2; y <= y1; y++)
                        {
                          x = x2 + (int )(grad*(double )(y - y2));
                          for (offset=-width/2; offset < (width+1)/2; offset++)
                            {
                              if (((x+offset) >= 0) && ((x+offset) < output.getWidth()))
                                {
                                  if ((output.getVal(x+offset,y) == 255))
                                    {
                                      num_pixels++;
                                    }
                                }
                            }
                        }
                    }
            }
        }   /* end of 'if ((x1 >= 0) && (x1 < aImage->x) &&...' */
      x1 = x2;
      y1 = y2;
    }


  return( num_pixels );


}   /* count_number_of_pixels_near_ellipse() */

//Finds Ellipses in an image
//going to use the randomized hough transform to do this
void houghEllipse(Image<byte> inputImage, int threshold, Image< PixRGB<byte> > &output) {

  Point2D<int> randomPixels[3];
  double pointX[3][25];//the neighborhood pixels of the 3 chosen random "edge"pixels
  double pointY[3][25];
  float slopeTan[3];
  float InterTan[3];
  float slopeCenterLine[2] = { 0.0F, 0.0F };
  float InterCenterLine[2] = { 0.0F, 0.0F };

  bool foundCenterPoint = true;
  bool foundOtherParams = true;

  int p, q;
  double  a,b,c,r1,r2,theta;
  double **matrixA;
  double *vectorB;
  double d;
  int* indx= (int *)malloc(4 * sizeof(int) );

  matrixA = alloc_array(3,3);
  vectorB = alloc_vector(3);

  list<ellipse> houghSpace;
  list<ellipse>::iterator Iter;
  int NUMBER=0;
  int Iterations = 0;

  int max = 0;
  list<ellipse>::iterator indexOfMax;
  list<ellipse> tempSpace;

   srand( time(NULL));
   while(Iterations < inputImage.getWidth() * inputImage.getHeight())// && NUMBER < 6000)
    {

      //      cout<<"iterations"<<Iterations<<endl;
      int pixelCount = 0;
      while(pixelCount != 3)//get the initial random numbers
        {
          randomPixels[pixelCount].i = rand()%320;
          randomPixels[pixelCount].j = rand()%240;
          if(inputImage.getVal(randomPixels[pixelCount].i, randomPixels[pixelCount].j) == 255)
            {
              if(pixelCount == 1)
                {
                  if(!(randomPixels[pixelCount].i == randomPixels[0].i))
                    {
                      pixelCount++;
                      //inputImage.setVal(randomPixels[pixelCount-1].i,randomPixels[pixelCount-1].j,0);
                      //                      cout<<"chosen points"<<randomPixels[pixelCount-1].i<<" "<<randomPixels[pixelCount-1].j<<endl;
                    }

                }else if(pixelCount == 2)
                  {
                    if(!(randomPixels[pixelCount].i == randomPixels[0].i || randomPixels[pixelCount].i == randomPixels[1].i))
                    {
                      pixelCount++;
                      //inputImage.setVal(randomPixels[pixelCount-1].i,randomPixels[pixelCount-1].j,0);
                      //cout<<"chosen points"<<randomPixels[pixelCount-1].i<<" "<<randomPixels[pixelCount-1].j<<endl;
                    }
                  }
              else
                {
                  pixelCount++;
                  //inputImage.setVal(randomPixels[pixelCount-1].i,randomPixels[pixelCount-1].j,0);
                  //          cout<<"chosen points"<<randomPixels[pixelCount-1].i<<" "<<randomPixels[pixelCount-1].j<<endl;
                }

            }
          //          NUMBER++;
          //          if(NUMBER ==  1000)
          // break;
        }
      //      if(NUMBER == 1000)
      //        break;

      foundCenterPoint = true;
      foundOtherParams = true;

      //cout<<"after getting random pixels"<<endl;
      //find the tangent lines
      //get the neighboring pixels of each pixel
      int count[3] = {0,0,0};
      int tempI = 0, tempJ = 0;
      for(int j = 0; j < 3; j++)
        {
          for(int i = 0; i < 5; i++)
            for(int k = 0; k < 5; k++)
              {
                tempI = randomPixels[j].i + (i-2);
                tempJ = randomPixels[j].j + (k-2);
                if(tempI >=0 && tempI <320 && tempJ >= 0 && tempJ < 240){
                  if(inputImage.getVal(tempI,tempJ)==255 || (tempI == randomPixels[j].i && tempJ == randomPixels[j].j))
                    {
                      if(randomPixels[j].j != tempJ || (randomPixels[j].j == tempJ && randomPixels[j].i == tempI)){
                      pointX[j][count[j]] = tempI;
                      pointY[j][count[j]] = tempJ;
                      count[j]++;
                      }
                    }
                }

              }

        }
      //    cout<<"test2"<<endl;
      // solve for the slope of the tangent
      for(int i = 0; i < 3; i++){
        findTanLine(count[i], pointX[i], pointY[i], slopeTan[i], InterTan[i]);
        //cout<<"slopeTan is "<<slopeTan[i]<<endl;
      }

      //solve for the ellipse center
      for(int i = 0; i < 2; i++){
        if(!isnan(slopeTan[i]) && !isnan(slopeTan[i+1]))
          {
            slopeCenterLine[i] = ((slopeTan[i] *((InterTan[i+1]-InterTan[i])/(slopeTan[i]-slopeTan[i+1]))) + InterTan[i] - ((randomPixels[i].j+randomPixels[i+1].j)/2.0))/(((InterTan[i+1]-InterTan[i])/(slopeTan[i]-slopeTan[i+1])) - ((randomPixels[i].i+randomPixels[i+1].i)/2.0));
            InterCenterLine[i] = ((randomPixels[i].j + randomPixels[i+1].j)/2.0) - slopeCenterLine[i] * ((randomPixels[i].i + randomPixels[i+1].i)/2.0);
            if(isnan(slopeCenterLine[i]))
              {
                InterCenterLine[i] = (randomPixels[i].i + randomPixels[i+1].i)/2.0;
              }
          }
        else if(!isnan(slopeTan[i]) && isnan(slopeTan[i+1])) //one is vertical
          {
            slopeCenterLine[i] = ((slopeTan[i] * randomPixels[i+1].i + InterTan[i]) - ((randomPixels[i].j + randomPixels[i+1].j)/2.0)) / (randomPixels[i+1].i - ((randomPixels[i].i + randomPixels[i+1].i)/2.0));
            InterCenterLine[i] = ((randomPixels[i].j + randomPixels[i+1].j)/2.0) - slopeCenterLine[i] * ((randomPixels[i].i + randomPixels[i+1].i)/2.0);
            if(isnan(slopeCenterLine[i]))
              {
                InterCenterLine[i] = randomPixels[i].i; //store the x-intercept instead
              }
          }
        else if(!isnan(slopeTan[i+1]) && isnan(slopeTan[i]))//another one is vertical
          {
            slopeCenterLine[i] = ((slopeTan[i+1] * randomPixels[i].i + InterTan[i+1]) - ((randomPixels[i].j + randomPixels[i+1].j)/2.0)) / (randomPixels[i+1].i - ((randomPixels[i].i + randomPixels[i+1].i)/2.0));
            InterCenterLine[i] = ((randomPixels[i].j + randomPixels[i+1].j)/2.0) - slopeCenterLine[i] * ((randomPixels[i].i + randomPixels[i+1].i)/2.0);
            if(isnan(slopeCenterLine[i]))
              {
                InterCenterLine[i] =  ((randomPixels[i].i + randomPixels[i+1].i)/2.0);
              }
          }
        else //both are vertical lines?
          {
            foundCenterPoint = false;
          }

      }
      if(foundCenterPoint)
        {
          //          cout<<"slope center line is "<<slopeCenterLine[0]<<" "<<slopeCenterLine[1]<<endl;
          if(!isnan(slopeCenterLine[0]) && !isnan(slopeCenterLine[1]))
            {
              //The intersection of the 2 lines above is the center
              p = (int)((InterCenterLine[1]-InterCenterLine[0])/(slopeCenterLine[0]-slopeCenterLine[1]));
              q = (int)((slopeCenterLine[0]*p)+InterCenterLine[0]);
            }
          else if(!isnan(slopeCenterLine[0]) && isnan(slopeCenterLine[1]) ) // one of the slope is vertical
            {
              p = (int) InterCenterLine[1];
              q = (int) ((slopeCenterLine[0] * InterCenterLine[1]) + InterCenterLine[0]);
            }
          else if(!isnan(slopeCenterLine[1]) && isnan(slopeCenterLine[0]) )//same as above
            {
              p = (int) InterCenterLine[0];
              q = (int) ((slopeCenterLine[1] * InterCenterLine[0]) + InterCenterLine[1]);
            }
          else
            {
              p = -1;
              q = -1;
              //both lines are vertical what to do?
              foundCenterPoint = false;

            }
          if(p < 0 || p > 320 || q < 0 || q > 240)
            foundCenterPoint = false;
        }

      if(foundCenterPoint) // continue only if successfully found the center point of the ellipse
        {
          for(int i = 1; i < 4; i++)
            {
              //          cout<<"point "<<i<<" is ("<<randomPixels[i].i<<","<<randomPixels[i].j<<")"<<endl;
              matrixA[i][1] = (randomPixels[i-1].i - p) * (randomPixels[i-1].i- p);
              matrixA[i][2] =(randomPixels[i-1].i-p) * (randomPixels[i-1].j-q) * 2;
              matrixA[i][3] = (randomPixels[i-1].j-q) * (randomPixels[i-1].j-q);
              vectorB[i] = 1;
            }


          if (lu_decomposition(matrixA, 3, indx, &d) != 0)
            {
              foundOtherParams = false;
            }
          if(foundOtherParams)
            {
              lu_back_substitution(matrixA, 3, indx,vectorB);

              a = vectorB[1];
              b = vectorB[2];
              c = vectorB[3];
              //cout << "a is "<<a<<" b is "<<b<<" c is "<<c<<" sqrt(a^2+b^2)"<<sqrt(a*a+b*b)<<endl;
              if(a * c - b * b > 0)
                {
                  if(transform_ellipse_parameters(a, b, c,&r1, &r2, &theta))
                    {
                      //  double pixcnt = ((double)count_pixels(inputImage,p,q,r1,r2,theta))/( 3.14159 * ( 3*(r1+r2) - sqrt((r1+3*r2)*(3*r1+r2)) ));
                      // cout<<pixcnt<<endl;
                      //if(pixcnt > .5)
                                               if(true)
                        {
                          //cout<<" p is "<<p<<" q is "<<q<<endl;
                          //cout<<"found ellipse number so far is "<<houghSpace.size()<<" NUMBER IS"<<NUMBER<<endl;
                          bool found = false;
                          //ellipse
                          for(Iter = houghSpace.begin(); Iter != houghSpace.end(); Iter++)
                            {
                              if(( fabs((*Iter).x - p) < 10 && fabs((*Iter).y - q) < 10 && fabs((*Iter).r1 - r1) < 5 && fabs((*Iter).r2 - r2) < 5 && fabs((*Iter).theta - theta) < .4487))
                                {
                                  found = true;
                                  (*Iter).confidence++;
                                  if((*Iter).confidence > max)
                                    {
                                      max = (*Iter).confidence;
                                      indexOfMax = Iter;
                                    }
                                }
                            }
                          if(!found)
                            {
                              ellipse temp;
                              temp.x = p;
                              temp.y = q;
                              temp.r1 = r1;
                              temp.r2 = r2;
                              temp.theta = theta;
                              temp.confidence = 1;
                              houghSpace.push_back(temp);
                            }
                          NUMBER++;
                        }
                    }
                }
            }
        }
      Iterations++;

    }
   //  cout <<  ((double)count_pixels(inputImage,(*indexOfMax).x,(*indexOfMax).y,(*indexOfMax).r1,(*indexOfMax).r2,(*indexOfMax).theta))/(3* 3.14159 * ( 3*((*indexOfMax).r1+(*indexOfMax).r2) - sqrt(((*indexOfMax).r1+3*(*indexOfMax).r2)*(3*(*indexOfMax).r1+(*indexOfMax).r2)) ))<<endl;
   ellipse temp;
   temp.x = (*indexOfMax).x;
   temp.y = (*indexOfMax).y;
   temp.r1 = (*indexOfMax).r1;
   temp.r2 = (*indexOfMax).r2;
   temp.theta = (*indexOfMax).theta;
   temp.confidence = (*indexOfMax).confidence;
   cout<<temp.confidence<<endl;
   tempSpace.push_back(temp);
    drawEllipse(output, tempSpace);
  free(indx);

}








int main() {
  float sigma = .7;
  float tlow = 0.2;
  float thigh = .97;

  unsigned char *edge;

  char *dirfilename = NULL;


  ModelManager camManager("ColorTracker Tester");
  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(camManager, "COcam", "cocam"));

  camManager.addSubComponent(gb);

  camManager.loadConfig("camconfig.pmap");

  gb->setModelParamVal("FrameGrabberSubChan", 0);
  gb->setModelParamVal("FrameGrabberBrightness", 128);
  gb->setModelParamVal("FrameGrabberHue", 180);



  camManager.start();

  Image< PixRGB<byte> > cameraImage;
  Image<byte> grayImg;
  Image< PixRGB<byte> > houghImg(320,240,ZEROS) ;
  //Image<byte> houghImg;
  cameraImage = gb->readRGB();
  xwin = new XWindow(cameraImage.getDims());
  xwin2 = new XWindow(cameraImage.getDims());
  xwin3 = new XWindow(cameraImage.getDims());

  while(1) {

    printf(".");
    cameraImage = gb->readRGB();

    xwin->drawImage(cameraImage);
    grayImg = luminance(cameraImage);

    canny(grayImg.getArrayPtr(), grayImg.getHeight(), grayImg.getWidth(), sigma, tlow, thigh, &edge, dirfilename);
    Image<unsigned char> edgeImage(edge, grayImg.getWidth(), grayImg.getHeight());
    xwin2->drawImage(edgeImage);
    houghImg = edgeImage;

    std::vector <LineSegment2D> lines = houghTransform(edgeImage, PI/180, 1, 80, houghImg);
    //houghEllipse(edgeImage,0,houghImg);
     xwin3->drawImage(houghImg);


  }


}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // BEOSUB_TEST_HOUGH_C_DEFINED
