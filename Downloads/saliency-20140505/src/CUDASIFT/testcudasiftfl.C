/*!@file HMAX/test-hmax5.C Test Hmax class and compare to original code */

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
// Primary maintainer for this file: John McInerney <jmcinerney6@gmail.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDASIFT/testcudasiftfl.C $
// $Id: testcudasiftfl.C 15310 2012-06-01 02:29:24Z itti $
//

#include "Component/ModelManager.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/Rectangle.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Transforms.H"
#include "Image/Convolutions.H"
#include "Media/FrameSeries.H"
#include "nub/ref.h"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "SIFT/ScaleSpace.H"
#include "SIFT/VisualObject.H"
#include "SIFT/Keypoint.H"
#include "SIFT/VisualObjectDB.H"
//#include "CUDASIFT/CUDAVisualObjectDB.H"
#include "CUDASIFT/CUDAVisualObject.H"

#include "CUDASIFT/tpimageutil.h"
#include "CUDASIFT/tpimage.h"
#include "CUDASIFT/cudaImage.h"
#include "CUDASIFT/cudaSift.h"
#include "CUDASIFT/cudaSiftH.h" //This one is an addition and null

#include "Util/log.H"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <cstdio>
#include <dirent.h>


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

// ######################################################################
std::vector<std::string> readList(std::string inName)
{
  std::ifstream inFile;
  inFile.open(inName.c_str(),std::ios::in);
  if(!inFile){
    LFATAL("Unable to open image path list file: %s",inName.c_str());
  }
  std::string sLine;
  std::vector<std::string> fList;
  while (std::getline(inFile, sLine)) {
    std::cout << sLine << std::endl;
    fList.push_back(sLine);
  }
  LINFO("%" ZU " paths in the file\n", fList.size());
  LINFO("file list : \n");
  for (unsigned int i=0; i<fList.size(); i++)
    LINFO("\t%s", fList[i].c_str());
  inFile.close();
  return fList;
}

std::string matchObject(Image<float> &ima, VisualObjectDB& vdb, float &score)
{
  std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
#ifdef GPUSIFT
  rutz::shared_ptr<CUDAVisualObject>
    vo(new CUDAVisualObject("PIC", "PIC", ima,
                            Point2D<int>(-1,-1),
                            std::vector<float>(),
                            std::vector< rutz::shared_ptr<Keypoint> >(),
                            false,true));
#else
  rutz::shared_ptr<VisualObject>
    vo(new VisualObject("PIC", "PIC", ima,
                            Point2D<int>(-1,-1),
                            std::vector<float>(),
                            std::vector< rutz::shared_ptr<Keypoint> >(),
                            false,true));
#endif
  const uint nmatches = vdb.getObjectMatches(vo, matches, VOMA_SIMPLE,
                                             100U, //max objs to return
                                             0.5F, //keypoint distance score default 0.5F
                                             0.5F, //affine distance score default 0.5F
                                             1.0F, //minscore  default 1.0F
                                             3U,   //min # of keypoint match
                                             100U, //keypoint selection thershold
                                             false //sort by preattentive
                                             );
  score = 0;
  float avgScore = 0, affineAvgDist = 0;
  int nkeyp = 0;
  int objId = -1;
  if (nmatches > 0)
    {
      rutz::shared_ptr<VisualObject> obj; //so we will have a ref to the last matches obj
      rutz::shared_ptr<VisualObjectMatch> vom;
      for (unsigned int i = 0; i < 1; ++i) // Loops just once?
        {
          vom = matches[i];
          obj = vom->getVoTest();
          score = vom->getScore();
          nkeyp = vom->size();
          avgScore = vom->getKeypointAvgDist();
          affineAvgDist = vom->getAffineAvgDist();

          objId = atoi(obj->getName().c_str()+3);

          return obj->getName(); // Everything that follows is dead?
          LINFO("### Object match with '%s' score=%f ID:%i",
                obj->getName().c_str(), vom->getScore(), objId);

          //calculate the actual distance (location of keypoints) between
          //keypoints. If the same patch was found, then the distance should
          //be close to 0
          double dist = 0;
          for (int keyp=0; keyp<nkeyp; keyp++)
            {
              const KeypointMatch kpm = vom->getKeypointMatch(keyp);

              float refX = kpm.refkp->getX();
              float refY = kpm.refkp->getY();

              float tstX = kpm.tstkp->getX();
              float tstY = kpm.tstkp->getY();
              dist += (refX-tstX) * (refX-tstX);
              dist += (refY-tstY) * (refY-tstY);
            }

          //   printf("%i:%s %i %f %i %f %f %f\n", objNum, obj->getName().c_str(),
          //       nmatches, score, nkeyp, avgScore, affineAvgDist, sqrt(dist));

          //analizeImage();
        }

    }

  return std::string("nomatch");
}

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;

  ModelManager *mgr = new ModelManager("Test SIFT with Feature Learning");


  mgr->exportOptions(MC_RECURSE);

  // required arguments
  // <dir|list> <id> <outputfile>

  if (mgr->parseCommandLine( (const int)argc, (const char**)argv,
                            "<cudadev> <dir|list:images> <vdbfile>", 3, 3) == false)
    return 1;

  std::string images,devArg;
  std::string vdbFileName;

  std::string testPosName; // Directory where test images are

  devArg = mgr->getExtraArg(0);
  images = mgr->getExtraArg(1);
  vdbFileName = mgr->getExtraArg(2);

  //MemoryPolicy mp = GLOBAL_DEVICE_MEMORY;
  int dev = strtol(devArg.c_str(),NULL,0);
  std::cout << "device = " << dev << std::endl;
  cudaSetDevice(dev);

  LINFO("Loading db from %s\n", vdbFileName.c_str());
  VisualObjectDB vdb;
  vdb.loadFrom(vdbFileName,false);


  std::string::size_type dirArg=images.find("dir:",0);
  std::string::size_type listArg=images.find("list:",0);
  std::string imagename;
  std::string::size_type spos,dpos;

  if((dirArg == std::string::npos &&
      listArg == std::string::npos) ||
     (dirArg != 0 && listArg != 0)){
    LFATAL("images argument is in one of the following formats -  dir:<DIRNAME>  or  list:<LISTOFIMAGEPATHSFILE>");
    return EXIT_FAILURE;
  }
  if(dirArg == 0)
    images = images.substr(4);
  else
    images = images.substr(5);

  // Now we run if needed
  mgr->start();

  std::vector<std::string> imageFileNames;
  if(dirArg == 0)
    imageFileNames = readDir(images);
  else
    imageFileNames = readList(images);

  for(unsigned int imgInd=0;imgInd<imageFileNames.size();imgInd++){
    Image<float> inputf = Raster::ReadGray(imageFileNames[imgInd]);

    //Pick off image name from full path name
    spos = imageFileNames[imgInd].find_last_of('/');
    dpos = imageFileNames[imgInd].find_last_of('.');
    imagename = imageFileNames[imgInd].substr(spos+1,dpos-spos-1);

    std::cout << "imageFileNames[" << imgInd << "] = " << imageFileNames[imgInd] << std::endl;
    std::cout << "spos = " << spos << " ,dpos = " << dpos << std::endl;
    std::cout << "imagename = " << imagename << std::endl;

    float score = 0.0;
    std::string objName = matchObject(inputf, vdb, score);
    printf("objName=%s,score=%f\n",objName.c_str(),score);
  }

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
