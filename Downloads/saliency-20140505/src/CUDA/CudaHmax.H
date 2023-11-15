/*!@file CUDA/CudaHmax.H Riesenhuber & Poggio's HMAX model for object recognition */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaHmax.H $
// $Id: CudaHmax.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDAHMAX_H_DEFINED
#define CUDAHMAX_H_DEFINED

#include "CUDA/CudaImage.H"
#include <vector>

#include <cmath>
#include <string>
#include <vector>
#include <dirent.h>

//! Object recognition model inspired from Riesenhuber & Poggio's HMAX model
/*! See Riesenhuber & Poggio, Nature Neuroscience, 1999.
 */

class CudaHmax {
public:
  //! Default constructor. Need to call init() later.
  CudaHmax();

  //! Constructor
  CudaHmax(MemoryPolicy mp, int dev, const int nori, const std::vector<int>& spacess,
       const std::vector<int>& scaless, const int c1spaceol = 2,
       const bool angleflag = true, const float s2t = 1.0F,
       const float s2s = 1.0F, const float stdmin = 1.75F,
       const float stdstep = 0.5F, const int fsmin = 3, const int fsstep = 1);

  //! Initialization
  void init(MemoryPolicy mp, int dev, const int nori, const std::vector<int>& spacess,
            const std::vector<int>& scaless, const int c1spaceol = 2,
            const bool angleflag = true, const float s2t = 1.0F,
            const float s2s = 1.0F);

  void initFilters(const float stdmin, const float stdstep, const int fsmin, const int fsstep);


  //! Free internal memory
  void freeMem();

  //! Destructor
  ~CudaHmax();

  //! For utilize the operation : read a directory and pass the list of files
  std::vector<std::string> readDir(std::string dirName);
  std::vector<std::string> readList(std::string inName);

  //! Compute C2 responses from an input image
  CudaImage<float> getC2(const CudaImage<float>& input);
  void initC1(CudaImage<float> **&c1Res);
  void clearC1(CudaImage<float> **&c1Res);
  void printCorners(const char name[], const CudaImage<float>& cim, bool cond);
  void writeOutImage(const CudaImage<float>& cim,std::string & fName);
  // Calculate just the C1 level (must be public if you need to learn patches)
  void getC1(const CudaImage<float>& input, CudaImage<float>**& c1Res);
  void sumFilter(const CudaImage<float>& image, const float radius, CudaImage<float>& newImage);
  void sumFilter(const CudaImage<float>& image, const Rectangle& support, CudaImage<float>& newImage);

protected:
  bool initialized; //!< keep track of whether we have allocated some memory
  int nsb; //!< number of scale bands
  int ns;  //!< number of scales
  int no;  //!< number of orientations
  CudaImage<float> **filter; //!< low-level processing filters [ns][no]
  MemoryPolicy itsMp;
  int itsDev;

  int c1SpaceOL;  //!< spatial overlap in C1 cells
  bool angleFlag; //!< normalize convolution by image energy if true
  float s2Target; //!< center of S2 target
  float s2Sigma;  //!< size of S2 target

  std::vector<int> spaceSS;
  std::vector<int> scaleSS;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
