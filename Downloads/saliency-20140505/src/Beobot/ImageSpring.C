/*!@file Beobot/ImageSpring.C derived from the image template class; all the
  pixels are linked to theirs neighbors with springs */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/ImageSpring.C $
// $Id: ImageSpring.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Beobot/ImageSpring.H"

#include "Channels/Jet.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/IO.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Util/Assert.H"
#include "Util/MathFunctions.H"


// ######################################################################
template<class T>
void ImageSpring<T>::getStats()
{
  ASSERT(this->initialized());

  // initialize mean as a zero Jet, same size as the image elements
  // and initialize the others on the same model
  mean = this->getVal(0); mean *= 0.0f; weight = mean; // initialize Jet for weight
  stdev = mean; T mom2(mean);

  for (int n = 0; n < nbHeuristic; n++)
    {
      T randomPixel = this->getVal(randomUpToNotIncluding(this->getSize()));
      mean += randomPixel;
      inplaceSquare(randomPixel);
      mom2 += randomPixel;
    }
  mean /= (float)nbHeuristic;
  mom2 /= (float)nbHeuristic;
  T mean2(mean); inplaceSquare(mean2);

  // E( ( X-E(X) )^2) = E(X^2)-E(X)^2
  // we might have negative values because of precision issues
  stdev = mom2;
  stdev -= mean2;
  inplaceRectify(stdev);
  stdev = sqrt(stdev);
  weight = inverse(stdev, 1.0f);  // load weight values
}

// ######################################################################
template<class T>
void ImageSpring<T>::getStatsDist()
{
  ASSERT(this->initialized());
  meanDist = 0.0; double mom2 = 0.0;

  for (int n = 0; n < nbHeuristic; n++)
    {
      int randomIndex = randomUpToNotIncluding(this->getSize());
      T randomPixel = this->getVal(randomIndex);

      bool neighborDefined = false; int neighborNumber, neighborIndex;
      int attempt = 0;
      while (! neighborDefined && attempt++ < 10)
        {
          neighborNumber = randomUpToNotIncluding(nbNeighbors) + 1;
          neighborDefined =
            getNeighbor(randomIndex, neighborNumber, neighborIndex);
        }
      if (neighborDefined) {
        T randomNeighbor = this->getVal(neighborIndex);
        double d = distance(randomPixel, randomNeighbor, weight);
        meanDist += d; mom2 += squareOf(d);
      } else n --;
    }
  meanDist /= (double)nbHeuristic;
  mom2 /= (double)nbHeuristic;

  // E( ( X-E(X) )^2) = E(X^2)-E(X)^2
  stdevDist = sqrt(mom2 - squareOf(meanDist));
}

// ######################################################################
template<class T>
void ImageSpring<T>::initClustering(bool initPosMasses)
{
  // at the begining the masses and the pixel are at the same place
  if (initPosMasses)
    {
      for (int x = 0; x < this->getWidth(); x++)
        for (int y = 0; y < this->getHeight(); y++)
          setPos(Point2D<int>(x, y), float(x), float(y));
      memcpy(oldPosX, posX, this->getSize() * sizeof(float));
      memcpy(oldPosY, posY, this->getSize() * sizeof(float));
    }

  // initialize the image statistics & springs stiffnesses:
  getStats(); getStatsDist(); computeStiff();
}

// ######################################################################
template<class T>
void ImageSpring<T>::computeStiff(void)
{
  for (int currentIndex = 0; currentIndex < this->getSize(); currentIndex++)
    {
      Point2D<int> currentPoint;
      getXY(currentIndex, currentPoint);

      // the masses should always try to come back to the pixels
      // they come from (to ensure spatial coherence):
      setStiff(currentPoint, 0, 0.5f);

      T currval = this->getVal(currentIndex);

      for (int n = 1; n <= nbNeighbors; n++)
        {
          int neighborIndex;
          if (getNeighbor(currentIndex, n, neighborIndex))
            {
              float alikeness = distance(currval,this->getVal(neighborIndex),weight);

              // we normalize that;
              // note the - sign (alikeness high if distance small)
              alikeness = -2.0f * (alikeness - meanDist) / stdevDist;

              if (alikeness > 0.0f) setStiff(currentPoint, n, alikeness);
              else setStiff(currentPoint, n, 0.0);
            }
        }
    }
}

// ######################################################################
template<class T>
void ImageSpring<T>::computePos( const float dt )
{
  float *newPosX = new float[this->getSize()];
  float *newPosY = new float[this->getSize()];

  for (int x = 0; x < this->getWidth(); x++)
    for (int y = 0; y < this->getHeight(); y++)
      {
        int ci; getIndex(Point2D<int>(x, y), ci);

        // the position of the current mass
        float X = posX[ci], Y = posY[ci];

        // the force on the current mass; first from the grid:
        float fx = stiff[ci][0] * (float(x) - X);
        float fy = stiff[ci][0] * (float(y) - Y);

        for (int n = 1; n <= nbNeighbors; n++)
          {
            int neighborIndex;
            if (getNeighbor(ci, n, neighborIndex))
              {
                // the stiffness of the current spring
                float stif = stiff[ci][n];

                // update the force
                fx += stif * (posX[neighborIndex] - X);
                fy += stif * (posY[neighborIndex] - Y);
              }
          }

        // friction
        const float gamma = 1.0f;

        // update position according to equations
        newPosX[ci] = 2.0f * posX[ci] - oldPosX[ci] + dt * dt * fx
          - dt * gamma * (posX[ci] - oldPosX[ci]);
        newPosY[ci] = 2.0f * posY[ci] - oldPosY[ci] + dt * dt * fy
          - dt * gamma * (posY[ci] - oldPosY[ci]);
      }

  // update the position arrays:
  delete [] oldPosX; delete [] oldPosY; oldPosX = posX; oldPosY = posY;
  posX = newPosX; posY = newPosY;
}

// ######################################################################
template<class T>
float ImageSpring<T>::getDistanceMasses(const int idx1, const int idx2) const
{ return sqrt(squareOf(posX[idx1]-posX[idx2]) + squareOf(posY[idx1]-posY[idx2])); }

// ######################################################################
template <class T>
void ImageSpring<T>::getPositions(Image< PixRGB<byte> >& img, const int zoom)
{
  img.resize(this->getWidth() * zoom, this->getHeight() * zoom, true); // clear
  float maxstiff = 0.0;

  // first draw the springs:
  for (int i = 0; i < this->getSize(); i ++)
    {
      Point2D<int> pp(int(zoom * (posX[i] + 0.5)), int(zoom * (posY[i] + 0.5)));
      int ni;

      for (int n = 1; n <= nbNeighbors; n ++)
        if (getNeighbor(i, n, ni))
          {
            Point2D<int> np(int(zoom * (posX[ni]+0.5)), int(zoom * (posY[ni]+0.5)));
            drawLine(img, pp, np,
                     PixRGB<byte>(150.0f + 50.0f * stiff[i][n],
                                  100.0f * stiff[i][n],
                                  0));
            if (stiff[i][n] > maxstiff) maxstiff = stiff[i][n];
          }
    }
  //LDEBUG("Max stiffness = %f", maxstiff);

  // now draw the masses:
  PixRGB<byte> blue(0, 0, 255);
  for (int i = 0; i < this->getSize(); i ++)
    {
      Point2D<int> pp(int(zoom * (posX[i] + 0.5)), int(zoom * (posY[i] + 0.5)));
      drawDisk(img, pp, 2, blue);
    }
}

//######################################################################
template <class T>
void ImageSpring<T>::goGraph(Image<int32> &marked, const int currentIndex,
                             const int32 color, const int begin)
{
  // how many of my neighbors are already in the group 'color' ?
  int nbNeighborsSameColor = 0, neighborIndex;
  for (int n = 1; n <= nbNeighbors; n ++)
    if (getNeighbor(currentIndex, n, neighborIndex) &&
        marked.getVal(neighborIndex) == color)
      nbNeighborsSameColor ++;

  // if less than 5 and we are not just starting the algorithm
  // then forget it, I'm not really in this group
  if (begin <= 0 && nbNeighborsSameColor < 5) return;

  // Yes, I have many friends in this group, mark me as belonging there
  marked.setVal(currentIndex, color);

  // and check if my unmarked and spatially close neighbors are also
  // in the group or not.  We decrement begin because the algo is at
  // least partly initialized
  for (int n = 1; n <= nbNeighbors; n ++)
    if (getNeighbor(currentIndex, n, neighborIndex) &&
        marked.getVal(neighborIndex) == 0  &&
        getDistanceMasses(currentIndex, neighborIndex) < 1.0)
      goGraph(marked, neighborIndex, color, begin - 1);
}

//######################################################################
template <class T>
void ImageSpring<T>::getClusteredImage(const Image< PixRGB<byte> > &scene,
                                       Image< PixRGB<byte> > &clusteredImage,
                                       Point2D<int> &supposedTrackCentroid,
                                       const Point2D<int>& previousTrackCentroid)
{
  int preferedMeanX = 0, preferedMeanY = 0;

  Image<int32> marked(this->getDims(), ZEROS);
  Image< PixRGB<byte> > clustered(scene.getDims(), NO_INIT);

  int pX = scene.getWidth() / this->getWidth();
  int pY = scene.getHeight() / this->getHeight();

  PixRGB<byte> whitePixel(255, 255, 255);

  float highestScore = std::numeric_limits<float>::max();
  // to make sure that it will be initialized by first cluster

  // this color is not a color but merely a label
  // this is used in image 'marked', 0 being not marked
  int32 color = 0;

  for (int x = 0; x < this->getWidth(); x++ )
    for (int y = 0; y < this->getHeight(); y++ )
      {
        color ++;

        int meanX = 0, meanY = 0;

        if (marked.getVal(x, y) == 0) // pixel is NOT marked yet
          {
            // marks all connex pixels to color
            goGraph(marked, x + this->getWidth() * y, color);

            PixRGB<int32> meanPixel(0, 0, 0);
            PixRGB<byte> tmpPixel;
            int nbPixels = 0;

            // get mean pixel value on newly marked zone:
            for (int xp = 0; xp < this->getWidth(); xp ++  )
              for (int yp = 0; yp < this->getHeight(); yp ++)
                if (marked.getVal(xp, yp) == color) // this point just marked
                  for (int dx = 0; dx < pX; dx ++)
                    for (int dy = 0; dy < pY; dy ++)
                      {
                        scene.getVal(xp * pX + dx, yp * pY + dy, tmpPixel);
                        meanPixel += tmpPixel;
                        meanX += xp * pX + dx;
                        meanY += yp * pY + dy;
                        nbPixels ++;
                      }
            meanPixel /= nbPixels; meanX /= nbPixels; meanY /= nbPixels;

            // do not consider very small groups:
            bool ignoreGroup;
            if (nbPixels < scene.getWidth() * scene.getHeight() / 30)
              { meanPixel.set(0, 0, 0); ignoreGroup = true; }
            else
              ignoreGroup=false;

            // set all pixels of that zone in 'clustered'
            // to that value
            for (int xp = 0; xp < this->getWidth(); xp ++)
              for (int yp = 0; yp < this->getHeight(); yp ++)
                if (marked.getVal(xp, yp) == color) // this point just marked
                  for (int dx = 0; dx < pX; dx ++)
                    for (int dy = 0; dy < pY; dy++)
                      clustered.setVal(xp * pX + dx, yp * pY + dy, meanPixel);

            // draw the cross at center of gravity
            if (!ignoreGroup)
              drawCross(clustered, Point2D<int>(meanX, meanY), whitePixel);

#define LIKE_CENTER 1.0
#define LIKE_PREVIOUS 1.0
#define LIKE_BIG 1.0

            /*
              These defines define the behaviour of the algo which will
              determine which of the centroids is the track centroid.

              We minimize a cost function :

               LIKE_CENTER : importance of the centroid being close to
               the center bottom of the screen

               LIKE_PREVIOUS : ...
               the previous centroid

               LIKE_BIG : ... being the centroid of a bug cluster
            */

            float currentScore =
              LIKE_CENTER *
                sqrt( double( (float)squareOf(meanX - clustered.getWidth()/2) +
                              (float)squareOf(meanY - clustered.getHeight()) ) )
              / (float)clustered.getHeight()
              + LIKE_PREVIOUS *
                  sqrt( double( squareOf(meanX - previousTrackCentroid.i) +
                                squareOf(meanY - previousTrackCentroid.j) ) )
              / (float)clustered.getHeight()
              - LIKE_BIG * ( (float)nbPixels / ((float)clustered.getHeight()*
                                                (float)clustered.getWidth()));
            /* note: it's kind of messy but the idea is that all the
               components be more or less between 0.0 and 1.0 */

            if ((currentScore < highestScore) && (!ignoreGroup))
              // we have a new champion !
              {
                preferedMeanX = meanX; preferedMeanY = meanY;
                highestScore = currentScore;
              }

          }
      }

  // dashed line in the center of the screen
  int xp = clustered.getWidth() / 2;
  for(int yp = 0; yp < clustered.getHeight(); yp ++)
    if (yp % 3 == 0) clustered.setVal(xp, yp, whitePixel);

  // thicker cross at track center of gravity
  drawCross(clustered, Point2D<int>(preferedMeanX, preferedMeanY), whitePixel, 5, 2);

  // output of the function
  clusteredImage = clustered;
  supposedTrackCentroid.i = preferedMeanX;
  supposedTrackCentroid.j = preferedMeanY;
}

// ######################################################################
// Instantiate for float Jets:
template class ImageSpring< Jet<float> >;

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
