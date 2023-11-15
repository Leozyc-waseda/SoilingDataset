/*!@file AppDevices/test-audioGrab.C Test audio digitizer */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-audioGrab.C $
// $Id: test-audioGrab.C 14682 2011-04-05 01:11:05Z farhan $
//

#include "Audio/AudioWavFile.H"
#include "Component/ModelManager.H"
#include "Devices/AudioGrabber.H"
#include "Devices/AudioMixer.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Util/log.H"
#include "fstream"
#include <signal.h>

static bool goforever = true;  //!< Will turn false on interrupt signal
#define SCALE 4

//! Signal handler (e.g., for control-C)
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; }

int main(int argc, const char **argv)
{
  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  // Instantiate a ModelManager:
  ModelManager manager("Test Model for AudioGrabber Class");

  // Instantiate our various ModelComponents:
  nub::soft_ref<AudioMixer> mix(new AudioMixer(manager));
  manager.addSubComponent(mix);

  nub::soft_ref<AudioGrabber> agb(new AudioGrabber(manager));
  manager.addSubComponent(agb);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  std::ofstream outFile("bit8data.dat");

  // Let's get some of the option values, to configure our window:
  const uint nsamples = agb->getModelParamVal<uint>("AudioGrabberBufSamples");
  int nchan = agb -> getModelParamVal<uint>("AudioGrabberChans");
  
  XWindow win(Dims(nsamples, 256 / SCALE * nchan),
              -1, -1, "test-audioGrab window");
  Image<byte> ima(nsamples, 256 / SCALE * nchan, NO_INIT);

  if (agb->getModelParamVal<uint>("AudioGrabberBits") != 8U)
    LFATAL("Sorry, this app only supports 8-bit audio grabbing.");

  // let's get all our ModelComponent instances started:
  manager.start();

  std::vector<AudioBuffer<byte> > rec;
  int totalSamp =0;

  // main loop:
  while(goforever) {

    // grab a buffer:
    AudioBuffer<byte> buffer;
    agb->grab(buffer);
    rec.push_back(buffer);

    // clear image:
    ima.clear();

    // draw the buffer into the image:
    Point2D<int> p1, p2;
    for (uint i = 1; i < nsamples; i ++)
      {
        totalSamp++;
        p1.i = i - 1;
        p1.j = (255 - buffer.getVal(i - 1, 0)) / SCALE;
        p2.i = i;
        p2.j = (255 - buffer.getVal(i, 0)) / SCALE;
        outFile << p1.j << std::endl;

        drawLine(ima, p1, p2, byte(255));

        if (nchan > 1)
          {
            p1.i = i - 1;
            p1.j = (511 - buffer.getVal(i - 1, 1)) / SCALE;
            p2.i = i;
            p2.j = (511 - buffer.getVal(i, 1)) / SCALE;
            drawLine(ima, p1, p2, byte(255));
          }
      }

    // display the image:
    win.drawImage(ima);
  }

  LINFO("writing out audio file: testAudio.wav");
  LINFO("total samps recorded is %d", totalSamp);
  // save our current records
  writeAudioWavFile("testAudio.wav", rec);

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
