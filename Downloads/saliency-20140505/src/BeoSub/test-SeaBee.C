/*!@file BeoSub/test-SeaBee.C Test BeoSub submarine basic functions */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-SeaBee.C $
// $Id: test-SeaBee.C 7063 2006-08-29 18:26:55Z rjpeters $
//

#include "BeoSub/SeaBee.H"
#include "Component/ModelManager.H"
#include "SIFT/VisualObjectDB.H"

//need current estimation stuff
int main(const int argc, const char **argv)
{

  // instantiate a model manager:


  ModelManager manager("BeoSub Main Action");

  // Instantiate our various ModelComponents:
  nub::soft_ref<SeaBee> sub(new SeaBee(manager));
  manager.addSubComponent(sub);

  manager.start();
        sub->test();

  //sub->LookForRedLight();

  //sub->FollowPipeLine();
  //sub->ApproachPipeLine();
  //sub->TestBin(1);
  //sub->CenterBin();
  //sub->TaskB();

  /*
  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  LINFO("start");

  // wait for visual start:
  rutz::shared_ptr<VisualObjectDB> startdb(new VisualObjectDB());
  if (startdb->loadFrom("/home/vien/startdb/startdb.vdb"))
    {
      int count = 0; const int maxcount = 120;
      while(++count < maxcount)
        {
          LINFO("Waiting for start image... %d/%d", count, maxcount);

          Image< PixRGB<byte> > img = sub->grabImage(BEOSUBCAMDOWN);
          rutz::shared_ptr<VisualObject> vo(new VisualObject("Grab", "", img));
          std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
          const uint nmatches =
            startdb->getObjectMatches(vo, matches, VOMA_KDTREEBBF,
                                      5U, 0.5F, 0.5F, 0.4F, 3U, 7U);
          if (nmatches > 0)
            {
              LINFO("Found! Let's get started.");
              break;
            }
        }
    }
  else
    LERROR("Cannot load start DB -- no visual start");

 #if 0
  sub->turnAbs(127, true);
  #endif
  Angle my = 127;
  sub->turnRel(my-sub->getHeading());
  sub->useRotVelPID(true);
  // go through gate:
  LINFO("diving...");
  sub->useDepthPID(true);
  sub->diveAbs(1.4F, true);

  LINFO("advancing...");
  sub->advanceRel(25.0F, true); //GATE FIX
  sub->diveAbs(3.2);
  if(!sub->approachArea("src/BeoSub/database/down/taskAdown", BEOSUBCAMDOWN, 10.0)){
    if(!sub->approachArea("src/BeoSub/database/front/taskAfront", BEOSUBCAMFRONT, 10.0)){

    }
  }
  sub->TaskA();

  if(!sub->approachArea("src/BeoSub/database/down/taskBdown", BEOSUBCAMDOWN, 10.0)){
    if(!sub->approachArea("src/BeoSub/database/front/taskBfront", BEOSUBCAMFRONT, 10.0)){

    }
  }
  sub->TaskB();

  if(!sub->approachArea("src/BeoSub/database/down/taskCdown", BEOSUBCAMDOWN, 10.0)){
    if(!sub->approachArea("src/BeoSub/database/front/taskCfront", BEOSUBCAMFRONT, 10.0)){

    }
  }
  sub->TaskC();

  sub->diveAbs(-1.0);
  sleep(300);
  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
*/
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
