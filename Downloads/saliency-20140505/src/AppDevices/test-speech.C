/*!@file AppDevices/test-speech.C Test speech synth */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-speech.C $
// $Id: test-speech.C 8384 2007-05-13 16:16:54Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/SpeechSynth.H"
#include "Util/log.H"

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Speech");

  // Instantiate our various ModelComponents:
  nub::soft_ref<SpeechSynth> speechSynth(new SpeechSynth(manager));
  manager.addSubComponent(speechSynth);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  //load wave file
  speechSynth->sendCommand("(set! daisy (wave.load \"/home/elazary/sounds/daisy.wav\"))", -10, true);
  LINFO("Wav file loaded\n");

  speechSynth->sendCommand("(wave.play daisy)", 0, true);
  LINFO("Sending next test\n");
  bool ret = speechSynth->sayText("Shalom world.", 0);
  while (ret == false)
  {
    ret = speechSynth->sayText("Shalom world.", 0);
  }
  speechSynth->sayText("low", 5);
  speechSynth->sayText("medium", 3);
  speechSynth->sayText("important", 0);
  speechSynth->sayText("Again Shalom world.", 0);

  speechSynth->flushQueue();


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
