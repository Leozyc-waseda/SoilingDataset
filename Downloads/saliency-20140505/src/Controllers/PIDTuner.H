/*!@file GUI/PIDTuner.H  A utility to tune pid control */

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Controllers/PIDTuner.H $
// $Id: PIDTuner.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef PIDTuner_H_DEFINED
#define PIDTuner_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Component/ModelOptionDef.H"
#include "Transport/TransportOpts.H"
#include "Media/FrameSeries.H"
#include "rutz/shared_ptr.h"
#include "Util/Types.H"
#include "Util/WorkThreadServer.H"
#include "Util/JobWithSemaphore.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/DrawOps.H"
#include "GUI/PrefsWindow.H"
#include "GUI/XWinManaged.H"
#include "Image/Layout.H"
#include "Controllers/PID.H"


// ######################################################################
class PIDTuner : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  PIDTuner(OptionManager& mgr,
             PID<float>  &pid,
             Dims d = Dims(256,256),
             const std::string& descrName = "PIDTuner",
             const std::string& tagName = "PIDTuner");

  //! Destructor
  ~PIDTuner();

  void start2();

  void setupGUI(ModelComponent* comp, bool recurse = false);
  void update(); //update the window

  void startThread(nub::ref<OutputFrameSeries> &ofs);
  void stopThread();


  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

private:
  rutz::shared_ptr<WorkThreadServer> itsThreadServer;
  PID<float>& itsPID;
  Dims itsWinDims;
  Image<PixRGB<byte> > itsPIDImg;
  nub::soft_ref<OutputFrameSeries> itsOfs;

  //Graph vars
  Point2D<int> itsLastLoc;
  int itsCurrentX;

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
