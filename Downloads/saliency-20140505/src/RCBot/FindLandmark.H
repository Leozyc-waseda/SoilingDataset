/*!@file RCBot/FindLandmark.H find a landmark using SceneRec            */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/FindLandmark.H $
// $Id: FindLandmark.H 9412 2008-03-10 23:10:15Z farhan $
//

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Component/ModelManager.H"
#include "Image/CutPaste.H"     // for inplacePaste()

#include "Corba/Objects/SceneRecServerSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "Corba/CorbaUtil.H"

#ifndef FIND_LANDMARK_H
#define FIND_LANDMARK_H

static omni_mutex thLandmarkInfoMutex;
static omni_mutex landmarkImgMutex;
static omni_mutex findLandmarkMutex;
static omni_condition findLandmarkCond(&findLandmarkMutex);

// ######################################################################
class FindLandmark : public omni_thread
{
public:
  FindLandmark(SceneRecServer_var inSceneRec);
  ~FindLandmark();
  void run(void *ptr);
  int getLandmark(Point2D<int> &landmarkLoc);
  void setImg(Image<PixRGB<byte> > &inImg);

private:
  SceneRecServer_var sceneRec;
  Point2D<int> thLandmarkLoc;
  int currentLeg;
  Image<PixRGB<byte> > img;
};
#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
