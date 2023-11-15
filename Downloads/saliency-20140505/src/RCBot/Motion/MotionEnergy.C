/*!@file RCBot/Motion/MotionEnergy.C detect motion in an image stream   */
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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/Motion/MotionEnergy.C $
// $Id: MotionEnergy.C 13756 2010-08-04 21:57:32Z siagian $
//

#include "MotionEnergy.H"

// ######################################################################
// ##### MotionEnergyPyrBuilder Functions:
// ######################################################################

//debuging
//#define DEBUG_MotionEnergy

#ifdef DEBUG_MotionEnergy
#include "GUI/DebugWin.H"
#endif

// ######################################################################
template <class T>
MotionEnergyPyrBuilder<T>::MotionEnergyPyrBuilder(const PyramidType typ,
                                                  const float gabor_theta,
                                                  const float intens,
                                                  const int timeDomainSize,
                                                  const float magThreshold ) :
  PyrBuilder<T>(),
  itsPtype(typ),
  itsGaborAngle(gabor_theta),
  itsGaborIntens(intens),
  itsTimeDomainSize(timeDomainSize),
  itsMagThreshold(magThreshold)
{
  ASSERT(itsTimeDomainSize == 3 || itsTimeDomainSize == 5);
}

// ######################################################################
template <class T>
ImageSet<T> MotionEnergyPyrBuilder<T>::build(const Image<T>& image,
                                             const int firstlevel,
                                             const int depth,
                                             PyramidCache<T>* cache)
{
  ImageSet<T> result(depth);
  return result;
}

// ######################################################################
template <class T>
void MotionEnergyPyrBuilder<T>::updateMotion(const Image<T>& image,
                                             const int depth)
{

  // create a pyramid with the input image
  ImageSet<T> pyr = buildPyrGeneric(image, 0, depth, itsPtype,
                                    itsGaborAngle, itsGaborIntens);

  imgPyrQ.push_back(pyr); //build the time domain

  //the first frame (push the same frame itsTimeDomainSize-1 times)
  if (imgPyrQ.size() == 1){
    for(unsigned int i=0; i<itsTimeDomainSize-1; i++)
      imgPyrQ.push_back(pyr); //build the time domain
  }

  if (imgPyrQ.size() > itsTimeDomainSize)        //limit the image time queue
    imgPyrQ.pop_front();
}

// ######################################################################
// get the motion from the image by using sobel grad detector
// to detect edge orientations in the space and time
template <class T>
Image<float> MotionEnergyPyrBuilder<T>::buildHorizontalMotionLevel(int scale)
{
  // if we have too few imgs in the Q, just return an empty image:
  if (imgPyrQ.size() < itsTimeDomainSize)
    return Image<float>();

  int width  = imgPyrQ[0][scale].getWidth();
  int height = imgPyrQ[0][scale].getHeight();

  Image<float> motion(width, height, ZEROS); // motion place holder

  // to make the operation look nice and readable
  // the parenthesis around the x are needed
  // otherwise a bug happens when using Pix(0,i-1)
#define Pix(t,x) (*(imgT[t] + (y_pos*width)+(x)))

  // build the time domain pointers
  typename Image<T>::iterator imgT[itsTimeDomainSize];
  for (unsigned int i=0; i < itsTimeDomainSize; i++){
    imgT[i] = imgPyrQ[i][scale].beginw();
  }

  for (int y_pos = 0; y_pos < height; y_pos++)
    {
      // start and end before the template runs out
      for (unsigned int i = (itsTimeDomainSize/2); i < width-(itsTimeDomainSize/2); i++){
        // calculate the sobel Gx and Gy

        float Gx = 0, Gy = 0;
        // use a sobel 3x3 or 5x5. based on timedomainsize
        switch(itsTimeDomainSize){
          // 3x3 sobel operation
        case 3:
          Gx = -1*Pix(0,i-1) + 0 + 1*Pix(0,i+1) +
            -2*Pix(1,i-1) + 0 + 2*Pix(1,i+1) +
            -1*Pix(2,i-1) + 0 + 1*Pix(2,i+1);

          Gy = 1*Pix(0,i-1) +  2*Pix(0,i) +  1*Pix(0,i+1) +
            0*Pix(1,i-1) +  0*Pix(1,i) +  0*Pix(1,i+1) +
            -1*Pix(2,i-1) + -2*Pix(2,i) + -1*Pix(2,i+1);
          break;

          // 5x5 sobel operation
        case 5:
          Gx = -1*Pix(0,i-2) +  -2*Pix(0,i-1) +   0*Pix(0,i) +  2*Pix(0,i+1) +  1*Pix(0,i+2) +
            -4*Pix(1,i-2) +  -8*Pix(1,i-1) +   0*Pix(1,i) +  8*Pix(1,i+1) +  4*Pix(1,i+2) +
            -6*Pix(2,i-2) + -12*Pix(2,i-1) +   0*Pix(2,i) + 12*Pix(2,i+1) +  6*Pix(2,i+2) +
            -4*Pix(3,i-2) +  -8*Pix(3,i-1) +   0*Pix(3,i) +  8*Pix(3,i+1) +  4*Pix(3,i+2) +
            -1*Pix(4,i-2) +  -2*Pix(4,i-1) +   0*Pix(4,i) +  2*Pix(4,i+1) +  1*Pix(4,i+2);

          Gy =  1*Pix(0,i-2) +   4*Pix(0,i-1) +   6*Pix(0,i) +  4*Pix(0,i+1) +  1*Pix(0,i+2) +
            2*Pix(1,i-2) +   8*Pix(1,i-1) +  12*Pix(1,i) +  8*Pix(1,i+1) +  2*Pix(1,i+2) +
            0*Pix(2,i-2) +   0*Pix(2,i-1) +   0*Pix(2,i) +  0*Pix(2,i+1) +  0*Pix(2,i+2) +
            -2*Pix(3,i-2) +  -8*Pix(3,i-1) + -12*Pix(3,i) + -8*Pix(3,i+1) + -2*Pix(3,i+2) +
            -1*Pix(4,i-2) +  -4*Pix(4,i-1) +  -6*Pix(4,i) + -4*Pix(4,i+1) + -1*Pix(4,i+2);
        default:
          break;
        }

        double mag =  sqrt((Gx*Gx) + (Gy*Gy));
        double angle = 0; // this represents the vel of the motion. 90 no motion.

        if (mag > itsMagThreshold){        // try to not process motion as a result of noise
          // divide x by y and not y by x as in the orig sobel alg
          // this is to get the vector oriented perallel to the edge

          if (Gy == 0){ // watch for divide by 0
            angle = atan(0); // + M_PI/2;
          } else {
            angle = atan((double)(Gx/Gy)); // + (M_PI/2);
          }

          // flip angles to be at the top.
          // Could also be done by changing the convX and convY
          angle *= -1;

          // Have all the angles pointing in the positive direction by
          // flipping the angle if its less then 0
          if (angle < 0) angle += M_PI;

          // dismiss any angles that violates the motion
          if (angle > 0.15 && angle < M_PI-0.15) {
            motion.setVal(i, y_pos, (float)(angle) - (M_PI/2));
          } else {
            motion.setVal(i, y_pos,0);
          }
        } else {
          motion.setVal(i, y_pos, 0);
        }
      }
    }

  return motion;
}

// ######################################################################
// get the motion from the image by using sobel grad detector
// to detect edge orientations in the space and time
template <class T>
Image<float> MotionEnergyPyrBuilder<T>::buildVerticalMotionLevel(int scale)
{
  // if we have too few imgs in the Q, just return an empty image:
  if (imgPyrQ.size() < itsTimeDomainSize)
    return Image<float>();

  int width  = imgPyrQ[0][scale].getWidth();
  int height = imgPyrQ[0][scale].getHeight();

  Image<float> motion(width, height, ZEROS); // motion place holder

  // build the time domain pointers
  typename Image<T>::iterator imgT[itsTimeDomainSize];
  for (unsigned int i=0; i<itsTimeDomainSize; i++)
    {
      imgT[i] = imgPyrQ[i][scale].beginw(); 
    }

  for (int x_pos = 0; x_pos<width; x_pos++){

#ifdef DEBUG_MotionEnergy
    // build an image of one slice throught the time and y domain
    Image<byte> debugImg(itsTimeDomainSize, height, ZEROS);
#endif

    // start and end before the template runs out
    for (unsigned int i = (itsTimeDomainSize/2); i < height-(itsTimeDomainSize/2); i++)
      {
        // calculate the sobel Gx and Gy

        float Gx=0, Gy=0;

        // to make the operation look nice and readable
        // the parenthesis around the y are needed because when
        // otherwise a bug happends when using Pix2(0,i-1)
#define Pix2(t,y) (*(imgT[t] + ((y)*width)+x_pos))

#ifdef DEBUG_MotionEnergy
        // build the image
        for (uint j=0; j<itsTimeDomainSize; j++)
          {
            debugImg.setVal(j,i,Pix2(j,i));
          }
#endif
        // use a sobel 3x3 or 5x5. based on timedomainsize
        switch(itsTimeDomainSize){
          // 3x3 sobel operation
        case 3:
          Gx = -1*Pix2(0,i+1) + 0 + 1*Pix2(2,i+1)  +
               -2*Pix2(0,i)   + 0 + 2*Pix2(2,i)    +
               -1*Pix2(0,i-1) + 0 + 1*Pix2(2,i-1);
          
          Gy =  1*Pix2(0,i+1) +  2*Pix2(1,i+1) +  1*Pix2(2,i+1) +
                0*Pix2(0,i)   +  0*Pix2(1,i)   +  0*Pix2(2,i)   +
               -1*Pix2(0,i-1) + -2*Pix2(1,i-1) + -1*Pix2(2,i-1);

          break;

          // 5x5 sobel operation
        case 5:
          Gx = -1*Pix2(0,i+2) + -2*Pix2(1,i+2) +  0*Pix2(2,i+2) +  2*Pix2(3,i+2) +  1*Pix2(4,i+2) +
               -4*Pix2(0,i+1) + -8*Pix2(1,i+1) +  0*Pix2(2,i+1) +  8*Pix2(3,i+1) +  4*Pix2(4,i+1) +
               -6*Pix2(0,i)   + -12*Pix2(1,i)  +  0*Pix2(2,i)   + 12*Pix2(3,i)   +  6*Pix2(4,i)   +
               -4*Pix2(0,i-1) + -8*Pix2(1,i-1) +  0*Pix2(2,i-1) +  8*Pix2(3,i-1) +  4*Pix2(4,i-1) +
               -1*Pix2(0,i-2) + -2*Pix2(1,i-2) +  0*Pix2(2,i-2) +  2*Pix2(3,i-2) +  1*Pix2(4,i-2);

          Gy =  1*Pix2(0,i+2) +  4*Pix2(1,i+2) +   6*Pix2(2,i+2) +  4*Pix2(3,i+2) +   1*Pix2(4,i+2) +
                2*Pix2(0,i+1) +  8*Pix2(1,i+1) +  12*Pix2(2,i+1) +  8*Pix2(3,i+1) +   2*Pix2(4,i+1) +
                0*Pix2(0,i)   +  0*Pix2(1,i)   +   0*Pix2(2,i)   +  0*Pix2(3,i)   +   0*Pix2(4,i)   +
               -2*Pix2(0,i-1) + -8*Pix2(1,i-1) + -12*Pix2(2,i-1) + -8*Pix2(3,i-1) +  -2*Pix2(4,i-1) +
               -1*Pix2(0,i-2) + -4*Pix2(1,i-2) +  -6*Pix2(2,i-2) + -4*Pix2(3,i-2) +  -1*Pix2(4,i-2);
        default:
          break;
        }

      double mag =  sqrt((Gx*Gx) + (Gy*Gy));
      double angle = 0; // this represents the vel of the motion. 90 no motion.

      if (mag > itsMagThreshold){        // try to not process motion as a result of noise
        // divide x by y and not y by x as in the orig sobel alg
        // this is to get the vector oriented perallel to the edge

        if (Gy == 0){ // watch for divide by 0
          angle = atan(0); // + M_PI/2;
        } else {
          angle = atan((double)(Gx/Gy)); // + (M_PI/2);
        }

        angle *= -1; // flip angles to be at the top. Could also be done by changing the
        // convX and convY

        // Have all the angles pointing in the positive direction by
        // fliping the angle if its less then 0
        if (angle < 0) angle += M_PI;

        // dismiss any angles that violate the motion
        if (angle > 0.15 && angle < M_PI-0.15) {
          motion.setVal(x_pos, i, (float)(angle) - (M_PI/2));
        } else {
          motion.setVal(x_pos, i, 0);
        }
      } else {
        motion.setVal(x_pos, i, 0);
      }

    }
#ifdef DEBUG_MotionEnergy
    Image<PixRGB<byte> > img = toRGB(rescale(debugImg, 256, 256));
    SHOWIMG(img);
    //SHOWIMG(rescale(debugImg,Dims(256, 256)));
#endif
  }

  return motion;
}

// ######################################################################
template <class T>
ImageSet<float> MotionEnergyPyrBuilder<T>::buildHorizontalMotion()
{
  const int depth = imgPyrQ[0].size();

  ImageSet<float> result(depth);

  for (int scale = 0; scale < depth; ++scale)
    result[scale] = buildHorizontalMotionLevel(scale);

  return result;
}

// ######################################################################
// get the motion from the image by using sobel grad detector to detect edge orientations
// in the space and time
template <class T>
ImageSet<float> MotionEnergyPyrBuilder<T>::buildVerticalMotion()
{
  const int depth = imgPyrQ[0].size();

  ImageSet<float> result(depth);

  for (int scale = 0; scale < depth; scale++)
    result[scale] = buildVerticalMotionLevel(scale);

  return result;
}

// ######################################################################
template <class T>
float MotionEnergyPyrBuilder<T>::DrawVectors(Image<T> &img, Image<float> &motion)
{
  //TODO: should check the mag is the same size as dir

  Image<float>::const_iterator mag_ptr = motion.begin();
  Image<float>::const_iterator mag_stop = motion.end();

  int inx=0;
  int avg_i = 0;
  double avg_angle = 0;

  while (mag_ptr != mag_stop)
    {
      int y = inx/motion.getWidth();
      int x = inx - (y*motion.getWidth());

      if (*mag_ptr != 0) {
        avg_i++;
        //avg_angle += (*mag_ptr+M_PI/2);
        avg_angle += (*mag_ptr);

        int scale_x = x * (img.getWidth()/motion.getWidth());
        int scale_y = y * (img.getHeight()/motion.getHeight());
        drawLine(img, Point2D<int>(scale_x,scale_y),
                 Point2D<int>((int)(scale_x+25*cos((*mag_ptr))),
                         (int)(scale_y-25*sin((*mag_ptr)))),
                 (byte)0);
      }
      mag_ptr++;
      inx++;
    }

  if (avg_i > 0){
    int xi = img.getWidth()/2;
    int yi = img.getHeight()/2;

    drawLine(img,Point2D<int>(xi, yi),
             Point2D<int>((int)(xi+75*cos(avg_angle/avg_i)),
                     (int)(yi-75*sin(avg_angle/avg_i))),
             (byte)0, 3);
    return avg_angle/avg_i;
  }

  return -999;
}

// ######################################################################
template <class T>
MotionEnergyPyrBuilder<T>* MotionEnergyPyrBuilder<T>::clone() const
{ return new MotionEnergyPyrBuilder<T>(*this); }

// ######################################################################
template <class T>
void MotionEnergyPyrBuilder<T>::reset()
{
  // imgPyrQ.clear();
}

template class MotionEnergyPyrBuilder<byte>;

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
