/*!@file SeaBee/SubController.C  Control motors and pid */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/SubController.C $
// $Id: SubController.C 10794 2009-02-08 06:21:09Z itti $
//

#include "SeaBee/SubController.H"
#include "Component/ModelOptionDef.H"
#include "Devices/DeviceOpts.H"

        class ControllerLoopListener : public BeeSTEMListener
        {
                public:
                        ControllerLoopListener(nub::soft_ref<SubController> &sb)
                                : itsSubController(sb)
                        {}

                        virtual ~ControllerLoopListener() {}


                        virtual void run()
                        {
                                while(1)
                                {
                                        int heading = itsSubController->getHeading();
                                        LINFO("Heading %i", heading);
                                }
                        }

                        virtual const char* jobType() const
                        { return itsJobType.c_str(); }

                        virtual int priority() const
                        { return itsPriority; }


                private:
                        const SubController* itsSubController;
                        const int itsPriority;
                        const std::string itsJobType;

        };



// ######################################################################
SubController::SubController(OptionManager& mgr,
           const std::string& descrName,
           const std::string& tagName):
  ModelComponent(mgr, descrName, tagName),
        itsPitchPID(0.1, 0, 0, -100, 100),
        itsRollPID(0.1, 0, 0, -100, 100),
        itsHeadingPID(0.1, 0, 0, -100, 100),
        itsDepthPID(0.1, 0, 0, -100, 100)

{
        itsBeeStem = nub::soft_ref<BeeSTEM>(new BeeSTEM(mgr,"BeeSTEM", "BeeSTEM", "/dev/ttyS1"));
        addSubComponent(itsBeeStem);
}

void SubController::start2()
{

  LINFO("Starting controller thread");
  //start a worker thread
  itsThreadServer.reset(new WorkThreadServer("SubController",1)); //start a single worker thread
  itsThreadServer->setFlushBeforeStopping(false);

  rutz::shared_ptr<ControllerLoop> j(new ControllerLoop(this));
  itsThreadServer->enqueueJob(j);


}

// ######################################################################
SubController::~SubController()
{ }

// ######################################################################
int SubController::getHeading()
{


        return 10;

}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
