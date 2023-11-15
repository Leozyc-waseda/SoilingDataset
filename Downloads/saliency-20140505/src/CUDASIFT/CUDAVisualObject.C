/*!@file SIFT/VisualObject.C Visual Objects to be recognized */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDASIFT/CUDAVisualObject.C $
// $Id: CUDAVisualObject.C 14295 2010-12-02 20:02:32Z itti $
//

#include "CUDASIFT/CUDAVisualObject.H"
#include "SIFT/VisualObject.H"
#include "SIFT/ScaleSpace.H"
#include "Image/ColorOps.H"
//#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/MathOps.H"
//#include "Image/Pixels.H"
//#include "Raster/Raster.H"

#include "CUDASIFT/cudaImage.h"
#include "CUDASIFT/cudaSift.h"
#include "CUDASIFT/tpimageutil.h"
#include "CUDASIFT/tpimage.h"
#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>

#include <cctype>


// ######################################################################
void CUDAVisualObject:: computeKeypoints()
{
  CudaImage img1;
  // If we were given an image but no keypoints, let's extract them now:
  if(itsImage.initialized()) printf("Image initialized in computeKeypoints.\n");
  else printf("Image not initialized in computeKeypoints.\n");

  if(itsKeypoints.empty()) printf("Keypoints empty in computeKeypoints.\n");
  else printf("Keypoints not empty in computeKeypoints.\n");

  if (itsImage.initialized() && itsKeypoints.empty()){
      LDEBUG("%s: initializing ScaleSpace from %dx%d image...",
             itsName.c_str(), itsImage.getWidth(), itsImage.getHeight());

    Image <float> fimage = luminance(itsImage); // CUDA SIFT only works on monochrome
    Image <float> lum = luminance(fimage);
    SImage<float> limg(lum.getWidth(),lum.getHeight(),lum.getArrayPtr());
    //rescale(fimage, 256, 256);//Shouldn't be needed
    ReScale(limg, 1.0/256.0f);
    unsigned int w = limg.GetWidth();
    unsigned int h = limg.GetHeight();

    std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

    AllocCudaImage(&img1, w, h, w, false, true);
    img1.h_data = limg.GetData();
    Download(&img1);
    SiftData siftData;
    // There's a problem here!  The second parameter should be the initial
    // number of keypoints.  AddKeypoints should increase this as needed.
    // Setting this to 128 or 1024 seems to give somewhat reasonable results.
    // The keypoint prune for the Hough transform removes almost all the keypoints
    // for other powers of 2 including:
    // 1, 2, 4, 8, 16, 32, 64, 256, 512, 2048, 8192, 16384
    // Also,
    // the number of keypoints is different for the first run, running
    // through more than once seems to give consistent results...
    for (int j=0; j<0;j++){
    InitSiftData(&siftData, 128, true, true);
    ExtractSift(&siftData, &img1, 3, 3, 0.3f, 0.02f);
    //PrintSiftData(&siftData);
    printf("DERK siftData.numPts=%d, siftData.maxPts=%d\n",siftData.numPts,siftData.maxPts);
    FreeSiftData(&siftData);
    }

    InitSiftData(&siftData, 128, true, true);
    //Try a few different parameters for threshold.  Lowe says 0.03f.
    //ExtractSift(&siftData, &img1, 3, 3, 0.3f, 0.04);
    ExtractSift(&siftData, &img1, 3, 3, 0.3f, 0.015f);
    //PrintSiftData(&siftData); //print them out to compare
    printf("DERK2 siftData.numPts=%d, siftData.maxPts=%d\n",siftData.numPts,siftData.maxPts);

    for(int k=0;k<siftData.numPts;k++){
      //printf("siftData.h_data[k].xpos=%f,siftData.h_data[k].ypos=%f\n",
      //siftData.h_data[k].xpos,siftData.h_data[k].ypos);
      float xpos = siftData.h_data[k].xpos;  //itsX
      float ypos = siftData.h_data[k].ypos;  //itsY
      float scale = siftData.h_data[k].scale; //itsS
      float orientation = siftData.h_data[k].orientation;  //itsO
      float score = siftData.h_data[k].score; //itsM? not sure

      std::vector<byte> OriFV = std::vector<byte>();
      //std::vector<byte> ColFV = std::vector<byte>(); //no color

      //In CUDA SIFT this is a float 128 vector.  Convert to byte, so
      //unsigned?  Some of the members of the float vector are negative.
      for(int l=0; l<128; l++){ //Hard coded as 128 byte vector
        ASSERT(siftData.h_data[k].data[l] > 0.0 - 0.0001);
        ASSERT(siftData.h_data[k].data[l] < 1.0 + 0.0001);
        //float temp = siftData.h_data[k].data[l] * 255.0;
        byte b = (byte) (siftData.h_data[k].data[l] * 255); // Vector components are [0,255]
        //if(temp > 255) printf("Oops, data not [0,1), temp = %.20f, byte=%d\n",temp,b); //Just checking
        //if(temp < 0) printf("Oops, data not [0,1), temp = %.20f, byte=%d\n",temp,b); //Just checking
        //printf("float siftData.h_data[%d].data[%d]= %f\n",k,l,siftData.h_data[k].data[l]);
        //printf("byte siftData.h_data[%d].data[%d]= %d\n",k,l,b);
        OriFV.push_back(b);
        //ColFV.push_back(b); //No color at this time
      }

      //float ambiguity = siftData.h_data[k].ambiguity; //No analog in CPU SIFT
      //float edgeness  = siftData.h_data[k].edgeness;  //No analog in CPU SIFT

      rutz::shared_ptr<Keypoint> r =
        rutz::shared_ptr<Keypoint>(new Keypoint(OriFV, xpos, ypos, scale, orientation, score));
      itsKeypoints.push_back(r);
    }
    FreeSiftData(&siftData);
    img1.h_data=NULL; //Points to fimage, which is soon out of scope.
    FreeCudaImage(&img1);
  }
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
