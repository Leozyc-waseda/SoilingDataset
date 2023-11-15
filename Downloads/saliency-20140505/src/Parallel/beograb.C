/*!@file Parallel/beograb.C Grab video & save PNM frames on beowulf local disks (slave)
 */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/beograb.C $
// $Id: beograb.C 6430 2006-04-06 20:01:17Z rjpeters $
//

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Util/Assert.H"
#include "Util/Types.H"

#include <cstdlib>
#include <signal.h>
#include <time.h>
#include <unistd.h>

// path where to write frames:
#define OUTPATH "/home/tmp/1"

static bool goforever = true;
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; }

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Beograb - Slave");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beograb Slave", "BeograbSlave", false));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  initRandomNumbers();

  // various processing inits:
  TCPmessage rmsg;                 // message being received and to process

  // let's get all our ModelComponent instances started:
  manager.start();

  // wait for data and process it:
  while(goforever)
    {
      int32 rframe, raction, rnode = -1;  // receive from any node
      if (beo->receive(rnode, rmsg, rframe, raction, 3)) // wait up to 3ms
        {
          //LINFO("Frame %d, action %d from node %d", rframe, raction, rnode);
          // select processing branch based on frame number:
          switch(raction)
            {
            case BEO_INIT:       // ##############################
              {
                // ooops, someone wants to re-initialize us!
                // ok, we have nothing to cleanup for ourselves...
                LINFO("got re-init order.");
                // reinitialization of beowulf is handled automatically.
              }
              break;
            case 1:              // ##############################
              {
                // get the color image:
                Image<PixRGB<byte> > ima = rmsg.getElementColByteIma();

                // write frame out:
                Raster::WriteRGB(ima, sformat("%s/frame%06d.ppm", OUTPATH, rframe));

                LINFO("Saved frame %d", rframe);
              }
              break;
            default: // ##############################
              LERROR("Bogus action %d -- IGNORING.", raction);
              break;
            }
        }
    }

  // we got broken:
  manager.stop();
  exit(0);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
