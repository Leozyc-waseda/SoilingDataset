/*!@file SeaBee/DownwardVision.H Downward Vision Agent         */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/DownwardVision.H $
// $Id: DownwardVision.H 9976 2008-07-25 15:20:06Z beobot $
//
//////////////////////////////////////////////////////////////////////////

#ifndef __DOWN_VISION_H__
#define __DOWN_VISION_H__


#include "SensorAgent.H"
#include "AgentManager.H"
#include "SeaBee/PipeRecognizer.H"

class AgentManager;

class DownwardVisionAgent : public SensorAgent
{
 public:

  DownwardVisionAgent(OptionManager& mgr,
                      nub::soft_ref<AgentManager> ama,
                      const std::string& name = "DownwardVisionAgent");

 private:

  bool pickAndExecuteAnAction();

  rutz::shared_ptr<PipeRecognizer> itsPipeRecognizer;
  //rutz::shared_ptr<BeoSubCross> itsCrossRecognizer;

  void lookForPipe(rutz::shared_ptr<Image<PixRGB<byte> > > segImg, rutz::shared_ptr<Image<PixRGB<byte> > > currImg);
  void lookForBin(rutz::shared_ptr<Image<byte> > segImg);
  void lookForCross(rutz::shared_ptr<Image<byte> > segImg);
  void runSaliency();
  void runStereo();

  Image<PixRGB<byte> > itsCameraImage;
  uint itsFrameNumber;
  nub::soft_ref<AgentManager> itsAgentManager;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
