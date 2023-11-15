/*!@file src/Features/test-LocalBinaryPatterns.C */

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
// Primary maintainer for this file: Dan Parks <danielfp@usc.edu>
// $HeadURL$
// $Id$
//

#include "Component/ModelManager.H"
#include "Image/DrawOps.H"
#include "Image/Kernels.H"
#include "Image/CutPaste.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Raster/Raster.H"
#include "Media/FrameSeries.H"
#include "Util/Timer.H"
#include "Util/CpuTimer.H"
#include "Util/StringUtil.H"
#include "Features/LocalBinaryPatterns.H"
#include "Learn/LogLikelihoodClassifier.H"
#include "rutz/rand.h"
#include "rutz/trace.h"

#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <stdio.h>

#define TRAIN_WIDTH 160
#define TRAIN_HEIGHT 160
#define SAMPLE_WIDTH 160
#define SAMPLE_HEIGHT 160

#define TEST_SIZE 20

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Test LocalBinaryPatterns");

  // Create random number generator
  rutz::urand rgen(time((time_t*)0)+getpid());

  // Create log likelihood classifier and local binary patterns objects
  LogLikelihoodClassifier ll = LogLikelihoodClassifier(7);
  std::vector<LocalBinaryPatterns> lbp;
  //lbp.push_back(LocalBinaryPatterns(1,8,0,false,true));
  lbp.push_back(LocalBinaryPatterns(2,16,0,false,true));
  lbp.push_back(LocalBinaryPatterns(3,24,0,false,true));

  if (manager.parseCommandLine(
        (const int)argc, (const char**)argv, "<texture1file> ... <textureNfile>", 2, 200) == false)
    return 0;

  manager.start();

  uint numCategories = manager.numExtraArgs();
  std::vector<std::string> texFile;
  std::vector<Image<float> > tex;
  for(uint i=0;i<numCategories;i++)
    {
      texFile.push_back(manager.getExtraArg(i));
      // Load textures
      tex.push_back(Raster::ReadGray(texFile[i]));
    }

  const Dims trainDims = Dims(TRAIN_WIDTH,TRAIN_HEIGHT);
  for(uint idx=0;idx<numCategories;idx++)
    {
      // Take disjoint crops from left side of textures to train as models
      float tw = std::min(TRAIN_WIDTH,int(tex[idx].getWidth()/2.0));
      float th = std::min(TRAIN_HEIGHT,int(tex[idx].getHeight()));
      for(uint xp=0;xp<=uint(tex[idx].getWidth()/2.0-tw);xp+=tw)
	for(uint yp=0;yp<=uint(tex[idx].getHeight()-th);yp+=th)
	  {
	    LINFO("Adding crop for id[%u] at pos [%ux%u]",idx,xp,yp);
	    Image<float> samp = crop(tex[idx],Rectangle(Point2D<int>(xp,yp),trainDims));
	    for(uint o=0;o<lbp.size();o++)
	      lbp[o].addModel(toRGB(samp),idx+1);
	  }
    }

  // Build model
  std::vector<LocalBinaryPatterns::MapModelVector> allModels; 
  for(uint o=0;o<lbp.size();o++) 
    { 
      LINFO("Building variance model for LBP class [%u]",o);
      lbp[o].buildModels();
      allModels.push_back(lbp[o].getModels());
    }
  LocalBinaryPatterns::MapModelVector completeModel;
  lbp[0].combineModels(allModels,completeModel);
  ll.setModels(completeModel);

  int numCorrect=0;
  const Dims sampleDims = Dims(SAMPLE_WIDTH,SAMPLE_HEIGHT);
  // Take random crops from right side of texture to test models
  for(uint s=0;s<TEST_SIZE;s++)
    {
      // Pick a random texture
      int idx = rgen.idraw(numCategories);
      // Select crop from the right side of the textures to test
      float tw = std::min(SAMPLE_WIDTH,int(tex[idx].getWidth()/2.0));
      float th = std::min(SAMPLE_HEIGHT,int(tex[idx].getHeight()));
      int xp,yp;
      xp=rgen.idraw_range(tex[idx].getWidth()/2.0,int(tex[idx].getWidth()-tw));
      yp=rgen.idraw_range(0,int(tex[idx].getHeight()-th));
      Image<float> samp = crop(tex[idx],Rectangle(Point2D<int>(xp,yp),sampleDims));
      // Load crop
      std::vector<float> hist;
      for(uint o=0;o<lbp.size();o++)
	{
	  std::vector<float> tmpHist = lbp[o].createHistogram(samp);
	  hist.insert(hist.begin(),tmpHist.begin(),tmpHist.end());
	}
      int gtIdx = idx+1;
      std::map<int,double> pdf = ll.predictPDF(hist);
      std::map<int,double>::const_iterator piter;
      int predIdx=-1;
      double predMax = -std::numeric_limits<double>::max();
      printf("Class Probabilities: ");
      for(piter=pdf.begin();piter!=pdf.end();piter++)
        {
          printf("%d [%f], ",piter->first,piter->second);
          if(piter->second > predMax)
            {
              predMax = piter->second;
              predIdx = piter->first;
            }
        }
      printf("\n");
      LINFO("Index Ground Truth [%d], Predicted [%d]",gtIdx,predIdx);
      if(predIdx == gtIdx) numCorrect++;
    }
  LINFO("Test Accuracy %f, Random chance would be %f",float(numCorrect)/TEST_SIZE,1.0F/numCategories);
  manager.stop();

}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */



