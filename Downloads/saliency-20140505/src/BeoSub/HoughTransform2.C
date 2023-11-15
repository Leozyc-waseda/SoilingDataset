#ifndef HoughTransform2_C
#define HoughTransform2_C

#include "BeoSub/HoughTransform2.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
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
#include "BeoSub/CannyEdge.H"
#include "MBARI/Geometry2D.H"
using namespace std;


void findEndPoints(int p, float theta, Point2D<int> *intercept1, Point2D<int> *intercept2, Dims dimensions)
{
         int imageWidth = dimensions.w();
         int imageHeight = dimensions.h();

          //convert d and angle to a pair of endpoints
          intercept1->i = -1;
          intercept2->i = -1;
          int tempX0, tempX1, tempY0, tempY1;
          int cosTheta = (int) cos(theta);
          int sinTheta = (int) sin(theta);

          //Find the possible image border intersects
          tempX0 = (p) / cosTheta; //y=0
          tempX1 = (p-imageHeight*sinTheta)/cosTheta; //y=imageHeight
          tempY0 = (p / sinTheta); //x = 0
          tempY1 = (p - imageWidth * cosTheta) / sinTheta; //x = imageWidth

          //Decide which two of the four intersects are valid and store them
          if(tempX0 > 0 && tempX0 < imageWidth)
          {
              intercept1->i = tempX0;
              intercept1->j = 0;
          }
          if(tempX1 > 0 && tempX1 < imageWidth)
          {
            if(intercept1->i == -1) {
              intercept1->i = tempX1;
              intercept1->j = imageHeight;
            }
            else {
              intercept2->i = tempX1;
              intercept2->j = imageHeight;
            }
          }
          if(tempY0 > 0 && tempY0 < imageHeight)
          {
            if(intercept1->i == -1) {
              intercept1->i = 0;
              intercept1->j = tempY0;
            }
            else {
              intercept2->i = 0;
              intercept2->j = tempY0;
            }
          }
          if(tempY1 > 0 && tempY1 < imageHeight)
          {
            if(intercept1->i == -1) {
              intercept1->i = imageWidth;
              intercept1->j = tempY1;
            }
            else {
              intercept2->i = imageWidth;
              intercept2->j = tempY1;
            }
          }

}
//finds the line by using hough transform
//thetaRes is the resolution of each theta
//dRes is the resolution of the D
//returns the number of lines found
std::vector <LineSegment2D> houghTransform(Image<byte> &inputImage,
                                           float thetaRes,
                                           float dRes,
                                           int threshold,
                                           Image< PixRGB<byte> > &output)
{
  LDEBUG("Inside houghTransform");
  int imageWidth = inputImage.getWidth();
  int imageHeight = inputImage.getHeight();
  //get the total number of angles and P from the resolution of theta and P's
  int numAngle = (int) (360 / thetaRes); //in degrees
  int numP = (int) (((max(imageHeight, imageWidth) / 2) - 1) / dRes);


  //stores the lines found in the image
  std::vector <LineSegment2D> lines;

  //accumulator -> represents the hough space
  int accumulator[numP][numAngle];

  LDEBUG("Clearing accumulator");

  //fill accumulator with zeroes
  for(int i = 0; i<numP; i++)
    {
      for(int j = 0; j < numAngle; j++)
        {
          accumulator[i][j] = 0;
        }
    }

  // Normal Parameterization of a Line:
  // p = x * cos(theta) + y * sin(theta)
  //////////////////////////Mike/////////////////////////////

  LDEBUG("Accumulator is cleared");

  float theta;
  for(int p = 0; p < numP; p++)
    {
      for(int angle = 0; angle < numAngle; angle++)
        {
          theta = angle * thetaRes;

          //skip the thrid quadrant
          if(theta == 180)
            theta = 271;

          Point2D<int> intercept1, intercept2;

          findEndPoints(p, theta, &intercept1, &intercept2, Dims(imageWidth, imageHeight));

          //increment accumulator every time a white pixel lies on the
          //line plotted by bresenham algorithm using the endpoints found before

          int x0 = intercept1.i;
          int x1 = intercept1.j;
          int y0 = intercept2.i;
          int y1 = intercept2.j;

          bool steep = abs(y1 - y0) > abs(x1 - x0);
          if(steep)
            {
              swap(x0, y0);
              swap(x1, y1);
           }

          if(x0 > x1)
            {
              swap(x0, x1);
              swap(y0, y1);
            }

          int deltax = x1 - x0;
          int deltay = abs(y1 - y0);
          int error = -deltax / 2;
          int ystep;
          int y = y0;
          int x;

          if(y0 < y1)
            {
              ystep = 1;
            }
          else
            {
              ystep = -1;
            }

          for(x = x0; x < x1; x++)
            {
              if(steep)
                {
                  if(inputImage.getVal(y,x) == 255)
                    {
                      accumulator[p][angle]++;
                    }
                }
              else
                {
                  if(inputImage.getVal(x,y) == 255)
                    {
                      accumulator[p][angle]++;
                    }
                }

              error = error + deltay;

              if(error > 0)
                {
                  y = y + ystep;
                  error = error - deltax;
                }
            }

          for(int i = 0; i<numP; i++)
            {
              for(int j = 0; j < numAngle; j++)
                {
                  theta = j * thetaRes;
                  if(accumulator[i][j] > threshold && accumulator[i][j] > accumulator[i-1][j]
                     && accumulator[i][j] > accumulator[i+1][j] && accumulator[i][j] > accumulator[i][j-1]
                     && accumulator[i][j] > accumulator[i][j+1])//found a peak
                    {
                      Point2D<int> lineEndPoint1, lineEndPoint2;
                      findEndPoints(i , theta, &lineEndPoint1, &lineEndPoint2, Dims(imageWidth, imageHeight));

                      LineSegment2D theLine(lineEndPoint1, lineEndPoint2);

                      lines.push_back(theLine);
                    }
                }
            }

        }
    }

    return lines;
}


  //////////////////////////////////////////////////////////
  /*
        //equation of the line is Xcos(theta)+ysin(theta) = D
        // fill the accumulator
        //        printf("numD is %d\n",numD);
    for(int j = 0; j < inputImage.getHeight(); j++ ) {
      for(int i = 0; i < inputImage.getWidth(); i++ )
      {
        if( inputImage.getVal(i,j) != 0 )//point is an edge
          for(int n = 0; n < numangle; n++ )//get all possible D and theta that can fit to that point
        {
          r =  (int)(i * cos(n*thetaRes) + j * sin(n*thetaRes));
          r += (numD -1)/2;

          if(r > 0 && r < numD) //Avoid segfaults
            accumulator[r][n]++;


        }
      }
    }

    Point2D<int> p1;
    Point2D<int> p2;
    Point2D<int> storage[inputImage.getWidth()*inputImage.getHeight()];
    double a, b;
    int totalLines = 0;
    int pointCount = 0;



    // find the peaks, ie any number greater than the threshold is considered a line
    for(int i = 0; i<numD; i++)
      {
        for(int j = 0; j < numangle; j++)
          {
            if(accumulator[i][j] > threshold && accumulator[i][j]> accumulator[i-1][j]
               && accumulator[i][j]> accumulator[i+1][j] && accumulator[i][j]> accumulator[i][j-1]
               && accumulator[i][j]> accumulator[i][j+1])//found a peak
              {
                totalLines++;

                //get the image coordinates for the 2 endpoint of the line
                a = cos(j*thetaRes);
                b = sin(j*thetaRes);

                //TODO: ROOM FOR IMPROVEMENT!
                for(int x = 0; x < inputImage.getWidth(); x++) {
                  for(int y = 0; y < inputImage.getHeight(); y++) {
                    if( inputImage.getVal(x,y) != 0 )//point is an edge
                      if(((int)(x * cos(j*thetaRes) + y * sin(j*thetaRes))) + ((numD -1)/2) == i)
                        {
                          storage[pointCount].i = x;
                          storage[pointCount].j = y;
                          pointCount++;
                        }
                  }
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
                for(int x=0; x < numtoDiscard; x++) {
                  for(int y = 0; y < pointCount; y++) {
                    if(discardPoints[x] != y){
                      p1.i = p2.i = storage[y].i;
                      p1.j = p2.j = storage[y].j;

                    }

                  }
                }
                //find the 2 endpoints of line
                //check if vertical
                if(fabs(a) < .001)
                  {

                    //search the 2 endpoints
                    for(int x = 0; x < pointCount; x++)
                      {

                        bool discard = false;
                        for(int k = 0; k < numtoDiscard; k++) {
                          if(discardPoints[k] == x)
                            discard = true;
                        }

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
                        for(int k = 0; k < numtoDiscard; k++) {
                          if(discardPoints[k] == x)
                            discard = true;

                        }

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


                if (thisLine.isValid()) {


                  lines.push_back(thisLine);
                  drawLine(output, p1, p2,  PixRGB<byte> (255,0,0), 1);
                }

              }
          }
          }*/

/*
functions to find the best fit line
*/



#endif

