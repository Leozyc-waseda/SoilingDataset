/*!@file Beobot/BeobotLauncher.C File template for developing a new class */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotLauncher.C $
// $Id: BeobotLauncher.C 13712 2010-07-28 21:00:40Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/BeoChip.H"

#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

//! Description of applications which we handle:
struct BeobotLauncherApp {
  char *descrip1, *descrip2;  // 20-char max each (one LCD line)
  char *name;                 // name to display at launch
  char *cmd;                  // command to run
};

//! List of applications which we handle
BeobotLauncherApp apps[100];
int napps = 0;

//! Our own little BeoChipListener
class BeobotLauncherListener : public BeoChipListener
{
public:
  BeobotLauncherListener(nub::soft_ref<BeoChip> beoc) :
    keepgoing(true), action(-1), idx(0), beochip(beoc) { }

  virtual ~BeobotLauncherListener() { }

  virtual void event(const BeoChipEventType t, const int valint,
                     const float valfloat)
  {
    if (t == KBD)
      {
        if ((valint & 0x1) == 0 && idx > 0) // leftmost key pressed
          display(--idx);
        if ((valint & 0x10) == 0 && idx < napps-1) // rtmost key pressed
          display(++idx);
        if ((valint & 0x4) == 0) { // middle key pressed
          LINFO("Launching: %s", apps[idx].cmd);
          // Tell user about what we are launching:
          beochip->lcdClear();
          beochip->lcdPrintf(0, 0, "     Launching:     ");
          beochip->lcdPrintf(0, 2, "%s", apps[idx].name);
          usleep(800000);
          action = idx; // main program will take it from here
        }
      }
  }

  virtual void display(const int i)
  {
    beochip->lcdPrintf(0, 1, "%s", apps[i].descrip1);
    beochip->lcdPrintf(0, 2, "%s", apps[i].descrip2);
    beochip->lcdPrintf(0, 3, "%s   Start!   %s",
                       i > 0 ? "<<<<" : "    ",
                       i < napps-1 ? ">>>>" : "    ");
  }

  volatile bool keepgoing;
  volatile int action;
  volatile int idx;
  nub::soft_ref<BeoChip> beochip;
};

//! Launch various programs on the Beobot
/*! This launcher should be placed in a cron or init script so that it
  will permanently be running. To prevent multiple copies of it from
  running at the same time, we use an flock on a file in /tmp, as
  suggested on a linuxquestions.org post (so even if the launcher dies
  badly, which may leave the temp file behind, it will be unlocked and
  will not prevent a new instance from starting). Here we use the
  BeoChip to let the user decide what to run. Just before we launch
  and monitor a new program, we clear and release the BeoChip so that
  the launched program can use it. Once the launched program
  terminates, we take the BeoChip over again and show our launch
  menu. */
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // obtain a lock and check whether other instances of BeobotLauncher
  // are running; if so, abort:
  int fdlock; struct flock fl;
  fl.l_type = F_WRLCK; fl.l_whence = SEEK_SET;
  fl.l_start = 0; fl.l_len = 1;

  if ( (fdlock = open("/tmp/BeobotLauncher.lock",
                      O_WRONLY | O_CREAT, 0666)) == -1 )
    PLFATAL("Cannot open /tmp/BeobotLauncher.lock");
  if (fcntl(fdlock, F_SETLK, &fl) == -1)
    PLFATAL("Cannot lock /tmp/BeobotLauncher.lock");

  // load our config file (list of possible actions):
  char buf[1000];
  FILE *f = fopen("/etc/BeobotLauncher.conf", "r");
  if (f == NULL) PLFATAL("Cannot open /etc/BeobotLauncher.conf");
  while(fgets(buf, 999, f)) {
    if (buf[0] == '#') continue;
    char *data = new char[strlen(buf)+1];
    strcpy(data, buf);
    apps[napps].descrip1 = data; data[20] = '\0';
    apps[napps].descrip2 = &data[21]; data[41] = '\0';
    apps[napps].name = &data[42]; data[62] = '\0';
    apps[napps].cmd = &data[63];
    napps ++;
  }
  fclose(f);

  // instantiate a model manager:
  ModelManager manager("Beobot Launcher");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev>", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  b->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));

  // let's register our listener:
  rutz::shared_ptr<BeobotLauncherListener> lis(new BeobotLauncherListener(b));
  rutz::shared_ptr<BeoChipListener> lis2; lis2.dynCastFrom(lis); // cast down
  b->setListener(lis2);
  bool blinkon = false;

  // let's get all our ModelComponent instances started:
  manager.start();
  b->lcdClear();
  b->lcdPrintf(0, 0, "-- BeobotLauncher --");
  b->debounceKeyboard(true);
  b->captureKeyboard(true);
  lis->display(0);

  // main loop:
  while(lis->keepgoing)
    {
      // wait for an action code or a termination condition while we
      // blink out tail lights:
      while(lis->action == -1 && lis->keepgoing == true)
        {
          if (blinkon)
            { b->setDigitalOut(2, false); b->setDigitalOut(3, false);
            blinkon = false; }
          else
            { b->setDigitalOut(2, true); b->setDigitalOut(3, true);
            blinkon = true; }

          usleep(250000);
        }

      // if it's an action, let's execute it:
      if (lis->action != -1)
        {
          // first stop the manager to release the beochip:
          b->lcdClear();
          b->setDigitalOut(2, false); b->setDigitalOut(3, false);
          manager.stop();
          lis->idx = 0;

          // now execute:
          if (system(apps[lis->action].cmd)) LERROR("Error in system()");

          // let's get all our ModelComponent instances re-started:
          manager.start();
          b->lcdClear();
          b->lcdPrintf(0, 0, "     Done with:     ");
          b->lcdPrintf(0, 2, "%s", apps[lis->action].name);

          usleep(800000);

          b->lcdClear();
          b->lcdPrintf(0, 0, "-- BeobotLauncher --");
          b->debounceKeyboard(true);
          b->captureKeyboard(true);
          lis->display(0);

          // done with this action:
          lis->action = -1;
          }
    }

  // release our lock:
  fl.l_type = F_UNLCK;
  if (fcntl(fdlock, F_SETLK, &fl) == -1)
    PLERROR("Error unlocking /tmp/BeobotLauncher.lock");
  if (close(fdlock) == -1)
    PLERROR("Error closing /tmp/BeobotLauncher.lock");
  if (unlink("/tmp/BeobotLauncher.lock") == -1)
    PLERROR("Error deleting /tmp/BeobotLauncher.lock");
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
