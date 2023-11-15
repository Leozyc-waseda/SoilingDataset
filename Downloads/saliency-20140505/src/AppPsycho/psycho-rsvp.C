/*!@file AppPsycho/psycho-rsvp.C Psychophysics display of rsvp images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-rsvp.C $
// $Id: psycho-rsvp.C 15232 2012-03-20 22:22:04Z pohetsn $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/ShapeOps.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Util/StringUtil.H"
#include "Util/sformat.H"

#include <fstream>

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Rsvp");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "filelist.txt", 1, -1)==false) return(1);

  // filelist.txt should contain 1 line per rsvp stream, each line contains (1) Y or N depending on whether there is a
  // target, comma, and (2) the sequence of image file names separated by commas.

  // hook our various babies up and do post-command-line configs:
  d->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  d->clearScreen();

  // Pre-load all the RSVP streams:
  std::vector<ImageSet<byte> > rsvp; std::vector<bool> target;
  std::ifstream ifs(manager.getExtraArg(0));
  if (ifs.is_open() == false) LFATAL("Could not open file: %s", manager.getExtraArg(0).c_str());
  while (ifs.eof() == false)
  {
    std::string line; std::getline(ifs, line);
    std::vector<std::string> fnames; split(line, ",", std::back_inserter(fnames)); if (fnames.size() < 2) continue;

    if (fnames[0] == "Y") target.push_back(true); else target.push_back(false);

    ImageSet<byte> iset;
    for (size_t i = 1; i < fnames.size(); ++i) iset.push_back(Raster::ReadGray(fnames[i]));

    rsvp.push_back(iset);
  }

  size_t nbseq = rsvp.size(); size_t index[nbseq]; for (size_t i = 0; i < nbseq; i ++) index[i] = i;
  randShuffle(index, nbseq);

  // main loop:
  int score = 0;
  for (size_t i = 0; i < nbseq; i ++)
    {
      // show a fixation cross on a blank screen:
      d->clearScreen();
      d->displayFixation();

      // ready to go whenever the user is ready:
      d->waitForKey();
      d->waitNextRequestedVsync(false, true);
      d->pushEvent(sformat("===== Showing sequence: %" ZU " =====", index[i]));

      // blink the fixation:
      d->displayFixationBlink();

      // Show the RVSP sequence:
      ImageSet<byte> & seq = rsvp[index[i]];
      LINFO("Showing sequence %zu with %u images...", index[i], seq.size());

      for (size_t k = 0; k < seq.size(); ++k)
      {
        d->displayImage(seq[k], false, PixRGB<byte>(0, 0, 0), k, true);
        for (int zzz = 0; zzz < 10; ++zzz) usleep(10000);//d->waitNextRequestedVsync(true, false);

      }
      // Done with the sequence:
      d->clearScreen();
      d->displayText("Target present (Y/N)?");

      // wait for key:
      bool done = false; char c; bool correct = false;
      do {
        c = char(d->waitForKey()); done = true;
        if (c == 'y' || c == 'Y' || c == '1')
        { if (target[index[i]]) correct = true; else correct = false; }
        else if (c == 'n' || c == 'N' || c == '0')
        { if (target[index[i]]) correct = false; else correct = true; }
        else done = false;
      } while (done == false);

      d->clearScreen();

      if (correct) { score += 10; d->displayText(sformat("Correct! Score = %d", score)); }
      else  { score -= 20; d->displayText(sformat("Wrong! Score = %d", score)); }

      d->pushEvent(sformat("--- Answer for sequence %" ZU " is %c correct= %d score= %d---",
                           index[i], c, correct, score));

      usleep(1000000);
    }

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

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
