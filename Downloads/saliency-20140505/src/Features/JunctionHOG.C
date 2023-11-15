/*!@file Features/JunctionHOG.C  */


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
// Primary maintainer for this file: Daniel Parks <danielfp@usc.edu>
// $HeadURL$
// $Id$
//

#include "Features/JunctionHOG.H"
#include "SIFT/FeatureVector.H"

JunctionHOG::JunctionHOG() :
  HistogramOfGradients(),
  itsNeighborDistance(0.0)
{
  
}

JunctionHOG::JunctionHOG(bool normalize, Dims cellDims, bool fixedDims, int numOrientations, bool signedOrient, int numContinuityBins, float neighborDistance) :
  HistogramOfGradients(normalize,cellDims,fixedDims,numOrientations,signedOrient),
  itsContinuityBins(numContinuityBins),
  itsNeighborDistance(neighborDistance)
{
  if(neighborDistance > 0)
    {
      for(int t=0;t<8;t++)
	{
	  float ang=float(t)/8.0*M_PI;
	  initializeNeighbors(itsRelevantNeighbors[t],itsPerpAngles[t],itsParallelAngles[t],ang);
	}
    }
}

JunctionHOG::~JunctionHOG() 
{
}


std::vector<float> JunctionHOG::createHistogramFromGradient(const Image<float>& gradmag, const Image<float>& gradang)
{
  std::vector<float> ret = HistogramOfGradients::createHistogramFromGradient(gradmag,gradang);
  std::vector<float> tmp = calculateJunctionHistogram(gradmag,gradang);
  ret.insert(ret.end(),tmp.begin(),tmp.end());
  return ret;
}


void JunctionHOG::initializeNeighbors(std::vector<Point2D<int> >& neighbors, std::vector<float>& perpAngles, std::vector<float>& parallelAngles, float angle)
{
  // NOTE: This makes the same assumptions about an image that Rectangle.H makes, namely that (0,0) is the top left of the image

  // Remember atan2 has coords inverted: atan2(y,x) 

  // Left center point
  float lcprot=atan2(0,-1)+angle;
  Point2D<int> lcp=Point2D<int>(round(cos(lcprot)*itsNeighborDistance),round(sin(lcprot)*itsNeighborDistance));
  float lca=0+angle; 
  lca=(lca>M_PI) ? lca-M_PI : lca;
  float lcaPerp=(lca>M_PI/2.0) ? lca-M_PI/2.0 : lca+M_PI/2.0;
  neighbors.push_back(lcp); parallelAngles.push_back(lca); perpAngles.push_back(lcaPerp);
  // Left upper point
  float luprot=atan2(-1,-1)+angle;
  Point2D<int> lup=Point2D<int>(round(cos(luprot)*itsNeighborDistance),round(sin(luprot)*itsNeighborDistance));
  float lua=M_PI/4.0+angle; 
  lua=(lua>M_PI) ? lua-M_PI : lua;
  float luaPerp=(lua>M_PI/2.0) ? lua-M_PI/2.0 : lua+M_PI/2.0;
  neighbors.push_back(lup); parallelAngles.push_back(lua); perpAngles.push_back(luaPerp);
  // Left lower point
  float llprot=atan2(1,-1)+angle;
  Point2D<int> llp = Point2D<int>(round(cos(llprot)*itsNeighborDistance),round(sin(llprot)*itsNeighborDistance));
  float lla=M_PI-M_PI/4.0+angle; 
  lla=(lla>M_PI) ? lla-M_PI : lla;
  float llaPerp=(lla>M_PI/2.0) ? lla-M_PI/2.0 : lla+M_PI/2.0;
  neighbors.push_back(llp); parallelAngles.push_back(lla); perpAngles.push_back(llaPerp);
  
  // Right center point
  float rcprot=atan2(0,1)+angle;
  Point2D<int> rcp=Point2D<int>(round(cos(rcprot)*itsNeighborDistance),round(sin(rcprot)*itsNeighborDistance));
  float rca=0+angle; 
  rca=(rca>M_PI) ? rca-M_PI : rca;
  float rcaPerp=(rca>M_PI/2.0) ? rca-M_PI/2.0 : rca+M_PI/2.0;
  neighbors.push_back(rcp); parallelAngles.push_back(rca); perpAngles.push_back(rcaPerp);
  // Right upper point
  float ruprot=atan2(-1,1)+angle;
  Point2D<int> rup=Point2D<int>(round(cos(ruprot)*itsNeighborDistance),round(sin(ruprot)*itsNeighborDistance));
  float rua=M_PI/4.0+angle; 
  rua=(rua>M_PI) ? rua-M_PI : rua;
  float ruaPerp=(rua>M_PI/2.0) ? rua-M_PI/2.0 : rua+M_PI/2.0;
  neighbors.push_back(rup); parallelAngles.push_back(rua); perpAngles.push_back(ruaPerp);
  // Right lower point
  float rlprot=atan2(1,1)+angle;
  Point2D<int> rlp = Point2D<int>(round(cos(rlprot)*itsNeighborDistance),round(sin(rlprot)*itsNeighborDistance));
  float rla=M_PI-M_PI/4.0+angle; 
  rla=(rla>M_PI) ? rla-M_PI : rla;
  float rlaPerp=(rla>M_PI/2.0) ? rla-M_PI/2.0 : rla+M_PI/2.0;
  neighbors.push_back(rlp); parallelAngles.push_back(rla); perpAngles.push_back(rlaPerp);

}


std::vector<float> JunctionHOG::calculateJunctionHistogram(Image<float> gradmag, Image<float> gradang)
{
  // Determine the number of cells, based on whether we have fixed dims or not
  Dims cells;
  if(itsFixedDims)
    cells = itsCellDims;
  else
    cells = Dims(int(round(float(gradmag.getWidth())/itsCellDims.w())),int(round(float(gradmag.getHeight())/itsCellDims.h())));

  const int w=gradmag.getWidth(), h=gradmag.getHeight();

  // Create a SIFT-like feature vector to soft bin the continuity values
  FeatureVector cntfv = FeatureVector(cells.w(),cells.h(),itsContinuityBins,true);
  Image<float>::const_iterator gmg = gradmag.begin(), gang = gradang.begin();

  // loop over rows:
  for (int i = 0; i < w; i ++)
    {
      // loop over columns:
      for (int j = 0; j < h; j ++)
	{
	  // Check for line continuity using unsigned angles into bins [0-7]
	  int oriBin=floor(((*gang < 0) ?  *gang + M_PI : *gang)/M_PI*8);
	  float continuityStrength=0;
	  float continuityParallel=0;
	  int validNeighbors = 0;
	  // Go through all relevant neighbors and accumulate continuity votes
	  for(size_t n=0;n<itsRelevantNeighbors[oriBin].size();n++)
	    {
	      Point2D<int> neigh = itsRelevantNeighbors[oriBin][n];
	      if(!gradmag.coordsOk(i+neigh.i,j+neigh.j))
		{
		  // Ignore this neighbor, it is outside of image bounds
		  continue;
		}
	      int offset=neigh.i*h+neigh.j;
	      // Convert neighbor orientation to unsigned orientation
	      float unsignedNeighAng=(gang[offset] < 0) ?  gang[offset] + M_PI : gang[offset];
	      // Vote strength of neighbor
	      continuityStrength += gmg[offset]; // gradmag.getValInterp(neigh);
	      // Add parallel support for continuity
	      float ctrParallel=itsParallelAngles[oriBin][n];
	      //LINFO("offset %d ctrParallel %f, uNO %f",offset,ctrParallel,unsignedNeighAng);
	      continuityParallel += cos(fabs(ctrParallel-unsignedNeighAng));
	      // Perpendicular support is negative parallel support
	      float ctrPerp=itsPerpAngles[oriBin][n];
	      continuityParallel -= cos(fabs(ctrPerp-unsignedNeighAng));
	      ASSERT(!isnan(continuityParallel) && !isnan(continuityStrength));
	      validNeighbors++;
	    }
	  // Normalize by the number of contributing neighbors
	  if(itsNormalize && validNeighbors>0)
	    {
	      continuityParallel/=float(validNeighbors);
	      continuityStrength/=float(validNeighbors);
	    }
	  float continuityAng = atan2(continuityParallel,continuityStrength);
	  float continuityMag = sqrt(continuityParallel*continuityParallel+continuityStrength*continuityStrength);

	  // Get bin fractions
	  const float xf = float(i)/float(w)*cells.w();
	  const float yf = float(j)/float(h)*cells.h();
	  // Add Orientation to bin (always unsigned orientation)
	  addToBin(xf,yf,continuityAng,continuityMag,true,cntfv);
	  gmg++;
	  gang++;
	}
    }
  // Return the continuity histogram
  
  std::vector<float> hist;
  if(itsNormalize)
    hist = normalizeFeatureVector(cntfv);
  else
    hist = cntfv.getFeatureVector();
  LINFO("Calculated Junction Histogram of size %Zu",hist.size());
  return hist;
}

