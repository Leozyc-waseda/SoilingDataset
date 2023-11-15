/*!@file BeoSub/BeoSubSaliency.H A class for quick-and-dirty saliency mapping */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubSaliency.H $
// $Id: BeoSubSaliency.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef BEOSUBSALIENCY_H_DEFINED
#define BEOSUBSALIENCY_H_DEFINED

#include "Component/ModelManager.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Util/Types.H"
#include "GUI/XWindow.H"
#include "Image/colorDefs.H"

#include <pthread.h>
#include <cstdlib>
#include <unistd.h>
#include <list>

// ######################################################################
// ##### Global options:
// ######################################################################

#define sml        2
#define delta_min  3
#define delta_max  4
#define level_min  0
#define level_max  2
#define maxdepth   (level_max + delta_max + 1)
#define normtyp    (VCXNORM_MAXNORM)

// relative feature weights:
#define IWEIGHT 0.7
#define CWEIGHT 1.0
#define OWEIGHT 1.0
#define FWEIGHT 4.0
#define SWEIGHT 2.0

// image size vars
#define IMAGEWIDTH 160
#define IMAGEHEIGHT 120

// action definitions CHANGE TO ENUM? FIX?
#define RETINA     1
#define WINNER     2
#define LUMINANCE  3
#define REDGREEN   4
#define BLUEYELLOW 5
#define ORI0       6
#define ORI45      7
#define ORI90      8
#define ORI135     9
#define CMAP       10
#define FLICKER    11
#define INTENSITY  12
#define SKINHUE    13

struct jobData{
  jobData() { };

  jobData(const int jt, const PyramidType pt, const float w,
          const float ori = 0.0F) :
    jobType(jt), ptyp(pt), weight(w), orientation(ori) { }

  int jobType;
  PyramidType ptyp;
  float weight;
  float orientation;
};

class BeoSubSaliency : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  BeoSubSaliency(OptionManager& mgr,
               const std::string& descrName = "SaliencyMT",
               const std::string& tagName = "SaliencyMT");

  //! Destructor
  ~BeoSubSaliency();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  //! Setup up with image to be processed and build the job queue
  Point2D<int> run(Image< PixRGB<byte> > img, bool debug = false);

  //! Compute a conspicuity map from an image
  void computeCMAP();
  //@}

protected:
  OModelParam<uint> itsNumThreads; //!< Number of worker threads to use

private:
  Image< PixRGB<byte> > colima;     // current color image
  Image<float> lum;     //curent luminance image
  Image<byte> r, g, b, y; //curent RGBY images
  Image<float> previma;  // previous float image
  Image<float> outmap;   // output saliency map
  Image<float> skinima;  //skin hue map

  // get prepared to grab, communicate, display, etc:

  Point2D<int> win, winsm; // coordinates of attended location
  float maxval;

  PyramidType ptyp;

  //xwin stuff
  rutz::shared_ptr<XWindow> wini, wino;
  bool debugmode;
  bool hasRun;

  //job stuff
  std::list<jobData> jobQueue;
  bool jobsReady;
  bool dataReady;
  int jobsDone;
  int totalJobs;
  bool gotRGBY; bool gotLum;

  //thread stuff
  pthread_t *worker;
  pthread_mutex_t jobLock, condLock, mapLock;
  pthread_cond_t jobCond;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
