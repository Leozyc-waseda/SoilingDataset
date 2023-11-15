/*!@file GUI/PIDTuner.C  A utility to tune pid controll */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Controllers/PIDTuner.C $
// $Id: PIDTuner.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Controllers/PIDTuner.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Image/CutPaste.H"
#include "GUI/DebugWin.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"

#include "Util/JobWithSemaphore.H"
#include "Util/StringUtil.H"
#include "Util/WorkThreadServer.H"
#include "Util/sformat.H"
#include "rutz/compat_snprintf.h"

#include <ctype.h>
#include <deque>
#include <iterator>
#include <stdlib.h> // for atoi(), malloc(), free()
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <vector>


namespace
{
  class PIDTunerLoop : public JobWithSemaphore
  {
    public:
      PIDTunerLoop(PIDTuner* pidGUI)
        :
          itsPIDTuner(pidGUI),
          itsPriority(1),
          itsJobType("GUI Loop")
    {}

      virtual ~PIDTunerLoop() {}

      virtual void run()
      {
        ASSERT(itsPIDTuner);
        while(1)
        {
          itsPIDTuner->update();
          usleep(10000);
        }
      }

      virtual const char* jobType() const
      { return itsJobType.c_str(); }

      virtual int priority() const
      { return itsPriority; }

    private:
      PIDTuner* itsPIDTuner;
      const int itsPriority;
      const std::string itsJobType;
  };
}


// ######################################################################
PIDTuner::PIDTuner(OptionManager& mgr,
           PID<float> &pid,
           Dims d,
           const std::string& descrName,
           const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  itsPID(pid),
  itsWinDims(d),
  itsLastLoc(0, d.h()/2),
  itsCurrentX(0)
{
  itsPIDImg = Image<PixRGB<byte> >(d,ZEROS);

}

// ######################################################################
PIDTuner::~PIDTuner()
{
}

void PIDTuner::start2()
{
}

void PIDTuner::startThread(nub::ref<OutputFrameSeries> &ofs)
{
  LINFO("Starting Gui thread");
  //start a worker thread
  itsThreadServer.reset(new WorkThreadServer("GuiThread",1)); //start a single worker thread
  itsThreadServer->setFlushBeforeStopping(false);
  rutz::shared_ptr<PIDTunerLoop> j(new PIDTunerLoop(this));
  itsThreadServer->enqueueJob(j);
  itsOfs = ofs;
}

void PIDTuner::stopThread()
{
 //TODO stop threads
}



void PIDTuner::update()
{

  float y = itsPID.getErr();

  Point2D<int> loc((int)itsCurrentX, itsPIDImg.getWidth()/2 + (int)(y*itsPIDImg.getHeight()/2));
  if (itsPIDImg.coordsOk(loc))
  {
    drawLine(itsPIDImg, itsLastLoc, loc, PixRGB<byte>(0,255,0));
    itsLastLoc = loc;
  }


  itsCurrentX++;
  if (itsCurrentX > itsPIDImg.getWidth())
  {
    itsPIDImg.clear();
    itsLastLoc = Point2D<int>(0, itsPIDImg.getHeight()/2);
    itsCurrentX = 0;
  }

  itsOfs->writeRGB(itsPIDImg, "PIDTuner GUI",
      FrameInfo("PIDTuner Display", SRC_POS));


}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
