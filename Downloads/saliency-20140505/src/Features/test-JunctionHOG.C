/*!@file src/Features/test-JunctionHOG.C */

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
#include "Features/JunctionHOG.H"
#include "Learn/SVMClassifier.H"
#include "rutz/rand.h"
#include "rutz/trace.h"

#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>

#define TRAIN_WIDTH 160
#define TRAIN_HEIGHT 160
#define SAMPLE_WIDTH 160
#define SAMPLE_HEIGHT 160

#define TEST_SIZE 20

std::vector<std::string> readDir(std::string inName)
{
        DIR *dp = opendir(inName.c_str());
        if(dp == NULL)
        {
          LFATAL("Directory does not exist %s",inName.c_str());
        }
        dirent *dirp;
        std::vector<std::string> fList;
        while ((dirp = readdir(dp)) != NULL ) {
                if (dirp->d_name[0] != '.')
                        fList.push_back(inName + '/' + std::string(dirp->d_name));
        }
        LINFO("%" ZU " files in the directory\n", fList.size());
        LINFO("file list : \n");
        for (unsigned int i=0; i<fList.size(); i++)
                LINFO("\t%s", fList[i].c_str());
        std::sort(fList.begin(),fList.end());
        return fList;
}



int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Test JunctionHOG");

  // Create random number generator
  rutz::urand rgen(time((time_t*)0)+getpid());

  // Create svm class so we can write out feature vectors
  SVMClassifier svm;

  if (manager.parseCommandLine(
        (const int)argc, (const char**)argv, "<usejunctions> <outputdir> <obj1dir> ... <objNdir>", 4, 20) == false)
    return 0;

  manager.start();



  uint numCategories = manager.numExtraArgs()-2;
  int useJunc = atoi(manager.getExtraArg(0).c_str()); 
  std::string outputDir = manager.getExtraArg(1);
  HistogramOfGradients *hog;
  bool normalizeHistogram = true;
  bool fixedHistogram = true; // if false, cell size fixed
  Dims cellSize = Dims(8,8); // if fixedHist is true, this is hist size, if false, this is cell size
  // Create Junction Histogram of Gradients Class or vanilla HOG
  if(useJunc == 1)
    {
      LINFO("Creating JunctionHOG class, useJunc %d, orig str %s",useJunc,manager.getExtraArg(0).c_str());
      hog = new JunctionHOG(normalizeHistogram,cellSize,fixedHistogram);
    }
  else
    {
      LINFO("Creating HistogramOfGradients class, useJunc %d, orig str %s",useJunc,manager.getExtraArg(0).c_str());
      hog = new HistogramOfGradients(normalizeHistogram,cellSize,fixedHistogram);
    }
    
  for(uint i=0;i<numCategories;i++)
    {      
      // Create output file
      uint argIdx = i+2;
      uint idx = i+1;
      std::string fName = outputDir;
      fName.append(sformat("/Obj%u.out",idx));
      std::string inputDir = manager.getExtraArg(argIdx);
      LINFO("Will append files from dir %s to output file %s",inputDir.c_str(),fName.c_str());
      std::vector<std::string> fileList = readDir(inputDir);
      // For each file in the directory, calculate a histogram
      for(uint f=0;f<fileList.size();f++)
	{
	  Image<PixRGB<byte> > img = Raster::ReadRGB(fileList[f]);
	  Image<float>  lum,rg,by;
	  getLAB(img, lum, rg, by);
	  std::vector<float> hist=hog->createHistogram(lum,rg,by);
	  svm.train(fName,idx,hist);
	}
    }
  LINFO("Use the files to test the classification performance of this feature vector");
  manager.stop();

}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */



