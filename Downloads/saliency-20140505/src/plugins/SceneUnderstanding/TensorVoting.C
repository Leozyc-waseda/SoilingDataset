/*!@file SceneUnderstanding/TensorVoting.C  */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/TensorVoting.C $
// $Id: TensorVoting.C 13878 2010-09-03 18:45:11Z lior $
//

#ifndef TensorVoting_C_DEFINED
#define TensorVoting_C_DEFINED

#include "plugins/SceneUnderstanding/TensorVoting.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
//#include "Image/OpenCVUtil.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Image/Convolutions.H"
#include "Simulation/SimEventQueue.H"
#include "GUI/DebugWin.H"
//#include "Image/OpenCVUtil.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

// ######################################################################
TensorVoting::TensorVoting() :
  itsSigma(2)
{
  createTensorFields(itsSigma);
  //generateImage();
}

void TensorVoting::createTensorFields(float sigma)
{
  //Create the tensor fields first, so we can save time latter
  itsBallField  = createBallTensorField(sigma);

  itsStickFields.clear();
  itsStickFields.resize(180);

  for(int i=0; i<180; i++)
  {
    float u = cos(i*M_PI/180);
    float v = sin(i*M_PI/180);
    TensorField stick = createStickTensorField(u, v, sigma);
    itsStickFields[i] = stick;
  }

}


void TensorVoting::generateImage()
{
  //This is used for testing
  inputImg = Image<float>(241, 521, ZEROS);
  for(uint i=0; i<inputImg.size(); i++)
    inputImg[i] = -1;
  inputImg.setVal(21,266,0.000000);
  inputImg.setVal(23,303,6.709837);
  inputImg.setVal(23,306,0.000000);
  inputImg.setVal(27,334,14.036243);
  inputImg.setVal(32,356,14.036243);
  inputImg.setVal(37,372,15.945396);
  inputImg.setVal(48,399,24.443954);
  inputImg.setVal(57,418,24.443954);
  inputImg.setVal(64,427,32.005383);
  inputImg.setVal(76,443,39.805573);
  inputImg.setVal(89,456,53.130100);
  inputImg.setVal(109,470,49.398705);
  inputImg.setVal(131,478,73.300758);
  inputImg.setVal(140,478,90.000000);
  inputImg.setVal(173,485,60.945396);
  inputImg.setVal(185,493,57.994617);
  inputImg.setVal(199,503,50.194427);
  inputImg.setVal(218,516,75.963753);
  inputImg.setVal(231,520,63.434948);
  inputImg.setVal(236,517,233.130096);
  inputImg.setVal(240,496,206.565048);
  inputImg.setVal(223,481,251.565048);
  inputImg.setVal(213,475,233.130096);
  inputImg.setVal(186,450,219.805573);
  inputImg.setVal(182,439,200.556046);
  inputImg.setVal(174,418,203.962494);
  inputImg.setVal(167,405,206.565048);
  inputImg.setVal(152,373,203.962494);
  inputImg.setVal(143,348,197.525574);
  inputImg.setVal(136,328,197.525574);
  inputImg.setVal(133,309,187.594650);
  inputImg.setVal(131,303,180.000000);
  inputImg.setVal(129,279,180.000000);
  inputImg.setVal(130,258,180.000000);
  inputImg.setVal(131,244,180.000000);
  inputImg.setVal(134,216,171.869904);
  inputImg.setVal(136,203,165.963760);
  inputImg.setVal(144,173,163.300751);
  inputImg.setVal(152,151,159.443954);
  inputImg.setVal(161,131,156.037506);
  inputImg.setVal(172,109,151.699249);
  inputImg.setVal(175,101,158.198593);
  inputImg.setVal(185,72,164.744888);
  inputImg.setVal(187,59,168.690063);
  inputImg.setVal(189,46,180.000000);
  inputImg.setVal(177,18,243.434952);
  inputImg.setVal(168,15,270.000000);
  inputImg.setVal(140,19,291.801422);
  inputImg.setVal(122,27,300.963745);
  inputImg.setVal(102,44,320.194427);
  inputImg.setVal(89,61,323.130096);
  inputImg.setVal(80,74,333.434937);
  inputImg.setVal(65,103,335.224854);
  inputImg.setVal(60,114,338.198578);
  inputImg.setVal(50,138,338.198578);
  inputImg.setVal(42,160,340.016907);
  inputImg.setVal(38,173,345.963745);
  inputImg.setVal(33,196,345.963745);
  inputImg.setVal(28,216,348.690063);
  inputImg.setVal(23,248,353.290161);
  inputImg.setVal(23,250,0.000000);


}

// ######################################################################
TensorVoting::~TensorVoting()
{
}


TensorField TensorVoting::transImage(Image<float>& img)
{

  TensorField tensorField(img.getDims(), ZEROS);

  for(int j=0; j<img.getHeight(); j++)
    for(int i=0; i<img.getWidth(); i++)
    {
      float val = img.getVal(i,j);
      if (val >= 0)
      {
          double x = cos((img.getVal(i)*M_PI/180) + (90*M_PI)/180);
          double y = sin((img.getVal(i)*M_PI/180) + (90*M_PI)/180);

          tensorField.t1.setVal(img.getWidth()+1-i, j, x*x);
          tensorField.t2.setVal(img.getWidth()+1-i, j, x*y);
          tensorField.t3.setVal(img.getWidth()+1-i, j, x*y);
          tensorField.t4.setVal(img.getWidth()+1-i, j, y*y);
      }
    }

  return tensorField;
}

TensorField TensorVoting::createStickTensorField(float u, float v, float sigma)
{
  // Calculate the window size from sigma using
  // equation 5.7 from Emerging Topics in Computer Vision
  // make the field odd, if it turns out to be even.
  double ws = floor( ceil(sqrt(-log(0.01)*sigma*sigma)*2) / 2 )*2 + 1;
  double wHalf = (ws-1)/2;

  //// Turn the unit vector into a rotation matrix
  //double btheta = atan2(v,u);


  //Generate our theta's at each point in the
  //field, adjust by our base theta so we rotate
  //in funcion. Also generate the attenuation field at the same time
  //This is taken from Equation
  //5.2 in Emerging Topics in Computer Vision. Note our
  //thetas must be symmetric over the Y axis for the arc
  //length to be correct so there's a bit of a coordinate
  //translation.

  Image<float> theta((int)ws, (int)ws, ZEROS);
  Image<float> DF((int)ws, (int)ws, ZEROS);
  double norm = sqrt(u*u + v*v);
  double uNorm = u/norm;
  double vNorm = v/norm;
  for(int j=0; j<ws; j++)
    for(int i=0; i<ws; i++)
    {
      double Zj = j-wHalf;
      double Zi = wHalf-i;
      double x = vNorm*Zi + uNorm*Zj;
      double y = uNorm*Zi - vNorm*Zj ;
      double th = atan2(y,x);
      theta.setVal(i,j, th);

      //The attenuation field
      th = fabs(th);
      if (th > M_PI/2)
        th = M_PI - th;
      th = 4*th; //This was not in the original spec

      double l = sqrt(x*x + y*y);

      double s = 0;
      if (l != 0 && th != 0)
        s = (th*l)/sin(th);
      else if (l==0 || th == 0)
        s = l;

      double k=0;
      if (l != 0)
        k = 2*sin(th)/l;

      double c= (-16*log2(0.1)*(sigma-1))/(M_PI*M_PI);
      double df = exp(-((s*s+c*(k*k))/(sigma*sigma)));
      if (th <= M_PI/2)
        DF.setVal(i,j,df);
    }

  //Generate the final tensor field attenuated
  TensorField stickField(theta.getDims(), NO_INIT);
  double bTheta = atan2(v,u);
  for(int j=0; j<theta.getHeight(); j++)
    for(int i=0; i<theta.getWidth(); i++)
    {
      double th = theta.getVal(i,j);
      double b1 = -sin(2*th + bTheta);
      double b2 = cos(2*th + bTheta);

      double att = DF.getVal(i,j);
      stickField.t1.setVal(i, j, b1*b1*att);
      stickField.t2.setVal(i, j, b1*b2*att);
      stickField.t3.setVal(i, j, b1*b2*att);
      stickField.t4.setVal(i, j, b2*b2*att);
    }

  return stickField;
}

TensorField TensorVoting::createBallTensorField(float sigma)
{

  //Create a ball tensor field by spinning a stick tensor
  double wsize = ceil(sqrt(-log(0.01)*sigma*sigma)*2);
  wsize = floor(wsize/2)*2+1;

  TensorField ballField(Dims((int)wsize, (int)wsize), ZEROS);
  for(float theta=0; theta < 2*M_PI; theta += (2*M_PI)/36)
  {
    float u = cos(theta); float v = sin(theta);

    TensorField stickField = createStickTensorField(u,v,sigma);
    ballField += stickField;
  }

  //Normalize the field
  ballField /= 36;

  EigenSpace eigen = getTensorEigen(ballField);

  return ballField;

}


void TensorVoting::getBallVotes(Image<float>& img,
    TensorField& tensorField, float sigma)
{

  //Calculate Ball voting Field
  //TensorField ballField = createBallTensorField(sigma);
  TensorField ballField = itsBallField;

  //Go through the image, and vote at given feature position
  for(int j=0; j<img.getHeight(); j++)
    for(int i=0; i<img.getWidth(); i++)
    {
      float val=img.getVal(i,j);

      if (val > 0)
      {
        //Go through the vote template and vote
        for(int y=0; y<ballField.t1.getHeight(); y++)
          for(int x=0; x<ballField.t1.getWidth(); x++)
          {
            int ii = i+x - (ballField.t1.getWidth()/2);
            int jj = j+y - (ballField.t1.getHeight()/2);

            if (tensorField.t1.coordsOk(ii,jj))
            {
              tensorField.t1.setVal(ii, jj,
                  tensorField.t1.getVal(ii,jj) +
                  ballField.t1.getVal(x,y)*val);

              tensorField.t2.setVal(ii, jj,
                  tensorField.t2.getVal(ii,jj) +
                  ballField.t2.getVal(x,y)*val);

              tensorField.t3.setVal(ii, jj,
                  tensorField.t3.getVal(ii,jj) +
                  ballField.t3.getVal(x,y)*val);

              tensorField.t4.setVal(ii, jj,
                  tensorField.t4.getVal(ii,jj) +
                  ballField.t4.getVal(x,y)*val);
            }

          }
      }
    }
}

TensorField TensorVoting::getStickVotes(const TensorField& tensorField,
                                        float sigma)
{

  //Calculate stick voting Field

  EigenSpace eigen = getTensorEigen(tensorField);

  TensorField voteField = tensorField;

  //Go thought the tensor, and vote at given feature position
  for(int j=0; j<eigen.l1.getHeight(); j++)
    for(int i=0; i<eigen.l1.getWidth(); i++)
    {
      //if the difference between the eigenvectors is greater the 0
      //then we have a stick vote

      float val=eigen.l1.getVal(i,j) - eigen.l2.getVal(i,j);
      if (val > 0)
      {
        //Get the direction of the vote from e1, while the weight is l1-l2
        float u = eigen.e1[1].getVal(i,j);
        float v = eigen.e1[0].getVal(i,j);

        //int angIdx = round((atan(u/v)-(M_PI/2))*180/M_PI);
        int angIdx = (int)round(atan(-u/v)*180/M_PI);
        if (angIdx < 0) angIdx += 180;

        //TensorField stickField = createStickTensorField(-u, v, sigma);
        TensorField stickField = itsStickFields.at(angIdx);

        //Go through the vote template and vote
        for(int y=0; y<stickField.t1.getHeight(); y++)
          for(int x=0; x<stickField.t1.getWidth(); x++)
          {
            int ii = i+x - (stickField.t1.getWidth()/2);
            int jj = j+y - (stickField.t1.getHeight()/2);

            if (voteField.t1.coordsOk(ii,jj))
            {
              voteField.t1.setVal(ii, jj,
                  voteField.t1.getVal(ii,jj) +
                  stickField.t1.getVal(x,y)*val);
              voteField.t2.setVal(ii, jj,
                  voteField.t2.getVal(ii,jj) +
                  stickField.t2.getVal(x,y)*val);
              voteField.t3.setVal(ii, jj,
                  voteField.t3.getVal(ii,jj) +
                  stickField.t3.getVal(x,y)*val);
              voteField.t4.setVal(ii, jj,
                  voteField.t4.getVal(ii,jj) +
                  stickField.t4.getVal(x,y)*val);
            }
          }
      }
    }

  return voteField;

}

TensorField TensorVoting::getStickVotes2(const TensorField& tensorField,
                                        float sigma)
{

  EigenSpace eigen = getTensorEigen(tensorField);

  TensorField voteField(tensorField.t1.getDims(), ZEROS);

  //Go thought the tensor, and vote at given feature position
  for(int j=0; j<eigen.l1.getHeight(); j++)
    for(int i=0; i<eigen.l1.getWidth(); i++)
    {
      //if the difference between the eigenvectors is greater the 0
      //then we have a stick vote

      float val=eigen.l1.getVal(i,j) - eigen.l2.getVal(i,j);
      if (val > 0)
      {
        //Get the direction of the vote from e1, while the weight is l1-l2
        float u = eigen.e1[1].getVal(i,j);
        float v = eigen.e1[0].getVal(i,j);

        //int angIdx = round((atan(u/v)-(M_PI/2))*180/M_PI);
        int angIdx = (int)round(atan(-u/v)*180/M_PI);
        if (angIdx < 0) angIdx += 180;

        //TensorField stickField = createStickTensorField(-u, v, sigma);
        TensorField stickField = itsStickFields.at(angIdx);

        //Go through the vote template and vote
        val=1;
        for(int y=0; y<stickField.t1.getHeight(); y++)
          for(int x=0; x<stickField.t1.getWidth(); x++)
          {
            int ii = i+x - (stickField.t1.getWidth()/2);
            int jj = j+y - (stickField.t1.getHeight()/2);

            if (voteField.t1.coordsOk(ii,jj))
            {
              voteField.t1.setVal(ii, jj,
                  voteField.t1.getVal(ii,jj) +
                  stickField.t1.getVal(x,y)*val);
              voteField.t2.setVal(ii, jj,
                  voteField.t2.getVal(ii,jj) +
                  stickField.t2.getVal(x,y)*val);
              voteField.t3.setVal(ii, jj,
                  voteField.t3.getVal(ii,jj) +
                  stickField.t3.getVal(x,y)*val);
              voteField.t4.setVal(ii, jj,
                  voteField.t4.getVal(ii,jj) +
                  stickField.t4.getVal(x,y)*val);
            }
          }
      }
    }

  return voteField;

}

TensorField TensorVoting::calcSparseField(Image<float>& img)
{

  TensorField tensorField(img.getDims(), ZEROS);

  for(uint i=0; i<img.size(); i++)
  {
    if (img[i] > 0)
    {
      tensorField.t1.setVal(i, 1);
      tensorField.t2.setVal(i, 0);
      tensorField.t3.setVal(i, 0);
      tensorField.t4.setVal(i, 1);
    }
  }

  return tensorField;

}

TensorField TensorVoting::calcRefinedField(TensorField& tensorField,
                                      Image<float>& img,
                                      float sigma)
{

  TensorField ballVoteField = tensorField;
  getBallVotes(img, ballVoteField, sigma);

  ////Erase anything that is not in the original image

  for(uint i=0; i<img.size(); i++)
    if (img[i] == 0)
    {
      ballVoteField.t1.setVal(i, 0);
      ballVoteField.t2.setVal(i, 0);
      ballVoteField.t3.setVal(i, 0);
      ballVoteField.t4.setVal(i, 0);
    }

  return (tensorField+ballVoteField);

}

TensorField TensorVoting::findFeatures(TensorField& tensorField, float sigma)
{

  EigenSpace eigen = getTensorEigen(tensorField);

 // Image<float> im = img;
 // //Normalize the gray scale image from 0 to 1;
 // float minVal, maxVal;
 // getMinMax(im, minVal, maxVal);
 // for(uint i=0; i<im.size(); i++)
 //   im[i] = im[i] / maxVal;


  //First step is to produce the initially encode the image
  //as sparse tensor tokens.
  TensorField sparseTf = calcSparseField(eigen.l1);

  LINFO("Refined field");
  TensorField refinedTf = calcRefinedField(sparseTf, eigen.l1, sigma);


  ////third run is to apply the stick tensor voting after
  ////zero'ing out the e2(l2) components so that everything
  ////is a stick vote.
  eigen = getTensorEigen(refinedTf);

  eigen.l2.clear();
  TensorField zeroTf = getTensor(eigen);

  LINFO("Stick Votes");
  tensorField = getStickVotes(zeroTf, sigma);
  LINFO("Done");

  return tensorField;

}

// ######################################################################
void TensorVoting::evolve()
{
  SHOWIMG(inputImg);
  TensorField tensorField = transImage(inputImg);


  tensorField = findFeatures(tensorField, itsSigma);

  //Show the features
  EigenSpace eigen = getTensorEigen(tensorField);
  Image<float> features = eigen.l1-eigen.l2;
  SHOWIMG(features);


}

TensorField TensorVoting::evolve(const Image<PixRGB<byte> >& img)
{

  Image<float> lum = luminance(img);
  SHOWIMG(lum);
  TensorField tensorField = getTensor(lum);

  //Extract tokens by keeping only the edges with values grater
  //then 10% of the max mag.
  Image<float> mag = getTensorMag(tensorField);
  float min, max;
  getMinMax(mag, min,max);

  for(uint i=0; i<mag.size(); i++)
    if (mag[i] < max*0.10)
      tensorField.setVal(i,0);

  tensorField = getStickVotes2(tensorField, itsSigma);

  return tensorField;

}

TensorField TensorVoting::evolve(const TensorField& tf, bool performNonMaxSurp)
{
  TensorField tensorField = tf;

  ////Extract tokens by keeping only the tensors with values grater
  ////then 10% of the max mag.
  //Image<float> mag = getTensorMag(tensorField);

  //float min, max;
  //getMinMax(mag, min,max);

  //for(uint i=0; i<mag.size(); i++)
  //  if (mag[i] < max*0.100)
  //    tensorField.setVal(i,0);

  itsTensorField = getStickVotes2(tensorField, itsSigma);

  if (performNonMaxSurp)
    nonMaxSurp(itsTensorField);


  return itsTensorField;

}

Image<float> TensorVoting::getTokensMag(bool normalize)
{
  EigenSpace eigen = getTensorEigen(itsTensorField);
  Image<float> tokens = eigen.l1-eigen.l2;

  if (normalize)
    inplaceNormalize(tokens, 0.0F, 255.0F);

  return tokens;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

