/*!@file FeatureMatching/OriChamferMatching.C Oriented chamfer matching algs */


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
// $HeadURL: $
// $Id: $
//

#ifndef ORICHAMFERMATCHING_C_DEFINED
#define ORICHAMFERMATCHING_C_DEFINED

#include "FeatureMatching/OriChamferMatching.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "GUI/DebugWin.H"
#include "Util/FastMathFunctions.H"

#include <stdio.h>

// ######################################################################
OriChamferMatching::OriChamferMatching() :
  itsUseOpenCV(false),
  itsMaxCost(100)
{
}

// ######################################################################
OriChamferMatching::OriChamferMatching(const std::vector<Line>& inLines,
    const int numDirections,
    const double oriCost,
    const Dims dims) :
  itsUseOpenCV(false),
  itsMaxCost(100)
{
  setLines(inLines, numDirections, oriCost, dims);
};

// ######################################################################
void OriChamferMatching::setLines(const std::vector<Line>& inLines,
    const int numDirections,
    const double oriCost,
    const Dims dims) 
{
  itsLines = inLines;
  //Construct the distance transform image

  //Build an image set with all the lines that fall in the quantized region
  itsOriDistImages = ImageSet<float>(numDirections, dims);

  for(uint i=0; i<inLines.size(); i++)
  {
    Line l = inLines[i];
    l.quantize(numDirections);

    int oriIdx = l.getDirectionIdx();
    if (oriIdx < 0 || oriIdx > numDirections)
      LFATAL("OriIdx %i is out of range %i", oriIdx, numDirections);
    //drawLine(itsOriDistImages[oriIdx], (Point2D<int>)l.getP1(),
    //    (Point2D<int>)l.getP2(), (float)l.getLength());
    drawLine(itsOriDistImages[oriIdx], (Point2D<int>)l.getP1(),
        (Point2D<int>)l.getP2(), 255.0F);
  }

  //Construct the distance images
  for(int oriIdx = 0; oriIdx < numDirections; oriIdx++)
  {
    if (itsUseOpenCV)
    {
      Image<byte> linesImg = binaryReverse(itsOriDistImages[oriIdx], 255.0F);
      linesImg.deepcopy();
      cvDistTransform(img2ipl(linesImg), img2ipl(itsOriDistImages[oriIdx]), CV_DIST_L2, 5);
    } else {
      //itsOriDistImages[oriIdx] = saliencyChamfer34(itsOriDistImages[oriIdx]);
      itsOriDistImages[oriIdx] = chamfer34(itsOriDistImages[oriIdx]);
    }

  }

  //Update the orientation costs
  updateOriCost(oriCost);
  buildIntegralDistances();
  
  //for(int oriIdx = 0; oriIdx < numDirections; oriIdx++)
  //{
  //  LINFO("oriIdx %i", oriIdx);
  //  SHOWIMG(itsOriDistImages[oriIdx]);
  //}

};

void OriChamferMatching::updateOriCost(const double oriCost)
{
  Dims d = itsOriDistImages[0].getDims();

  int size=d.w()*d.h();
  const int numDirections = itsOriDistImages.size();

  for(int k=0; k<size; k++) //For each pixel in the set
  {
    std::vector<float> costs(numDirections);
    //Assign the cost vector and Clamp all costs to maxCost
    for(uint i=0; i<costs.size(); i++)
    {
      costs[i] = itsOriDistImages[i][k];
      if (costs[i] > itsMaxCost)
        costs[i] = itsMaxCost;
    }

    //forward pass
    if (costs[0] > costs[numDirections-1] + oriCost)
      costs[0] = costs[numDirections-1] + oriCost;

    for (int i=1 ; i<numDirections; i++)
    {
      if (costs[i] > costs[i-1] + oriCost)
        costs[i] = costs[i-1] + oriCost;
    }

    if (costs[0] > costs[numDirections-1] + oriCost)
      costs[0] = costs[numDirections-1] + oriCost;

    for (int i=1 ; i<numDirections ; i++)
    {
      if (costs[i] > costs[i-1] + oriCost)
        costs[i] = costs[i-1] + oriCost;
      else
        break;
    }

    ////backward pass
    if (costs[numDirections-1] > costs[0] + oriCost)
      costs[numDirections-1] = costs[0] + oriCost;
    for (int i=numDirections-1 ; i>0 ; i--)
    {
      if (costs[i-1] > costs[i] + oriCost)
        costs[i-1] = costs[i] + oriCost;
    }

    if (costs[numDirections-1] > costs[0] + oriCost)
      costs[numDirections-1] = costs[0] + oriCost;
    for (int i=numDirections-1 ; i>0 ; i--)
    {
      if (costs[i-1] > costs[i] + oriCost)
        costs[i-1] = costs[i] + oriCost;
      else
        break;
    }

    //Assign the cost
    for (int i=0 ; i<numDirections ; i++)
      itsOriDistImages[i][k] = costs[i];

  }
}


void OriChamferMatching::buildIntegralDistances()
{
  int numDirections = itsOriDistImages.size();
  itsOriIntDistImages.resize(numDirections);

  for(int i=0; i<numDirections; i++)
  {
		double theta = (i*M_PI)/numDirections + M_PI/(2*numDirections);
    //printf("Theta %f\n", theta*180/M_PI);

    itsOriIntDistImages[i] = OriIntegralImage(itsOriDistImages[i], cos(theta), sin(theta));
  }
}


OriIntegralImage::OriIntegralImage(const Image<float>& distImage, float dx, float dy)
{
	if (fabs(dx) > fabs(dy))
	{
		itsDS = dy / (dx + 1e-9f);
		itsXindexed = 1;		
	}
	else
	{
		itsDS = dx / (dy + 1e-9f);
		itsXindexed = 0;
	}
	// Compute secant
	itsFactor = sqrt(itsDS*itsDS + 1);

  //Build the indecies used to find the pixel location
  int width = distImage.getWidth();
  int height = distImage.getHeight();

  if (itsXindexed)
    itsIndices.resize(width);
  else
    itsIndices.resize(height);

  for(uint i=0; i<itsIndices.size(); i++)
    itsIndices[i] = (int)ceil(i*itsDS - 0.5);

  itsIntegralImage = Image<float>(width, height, NO_INIT);

  //Build the integral image
  for(int x=0; x<width; x++)
    itsIntegralImage.setVal(x,0, 0.0F);

  for(int y=0; y<height; y++)
    itsIntegralImage.setVal(0,y, 0.0F);


  if (itsXindexed)
  {
    int miny=0, maxy=0;
    //Find the miny and maxy
    if (itsIndices[width-1] > 0)
    {
      miny = -itsIndices[width-1];
      maxy = height;
    } else {
      miny = 0;
      maxy = height - itsIndices[width-1];
    }

    //Build the integral image
    for(int y=miny; y<=maxy; y++)
      for(int x=1; x<width; x++)
      {
        int py = y + itsIndices[x-1];
        int cy = y + itsIndices[x];

        if (cy > 0 && cy < height - 1)
          itsIntegralImage.setVal(x, cy, 
              itsIntegralImage.getVal(x-1,py) + itsIntegralImage.getVal(x,cy));
      }
  } else {
    int minx =0, maxx = 0;
    //Find the minx and maxx
    if(itsIndices[height-1]>0)
    {
      minx = -itsIndices[height-1];
      maxx = width;
    } else {
      minx = 0;
      maxx = width - itsIndices[height-1];
    }

    //Build the integral image
    for(int x=minx; x<=maxx; x++)
      for(int y=1; y<height; y++)
      {
        int px = x + itsIndices[y-1];
        int cx = x + itsIndices[y];

        if (cx > 0 && cx < width - 1)
          itsIntegralImage.setVal(cx, y, 
              itsIntegralImage.getVal(px, y-1) + itsIntegralImage.getVal(cx,y));
      }
        
  }

}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

