/*!@file AppPsycho/psycho-classic-antisaccade.C Psychophysics display of pro-/anti-saccade task.
   The paradigm is similar to Munoz et al., 1998 except that targets are 10 degree eccentric and 
   only overlap condition is used. 
   ./bin/psycho-classic-antisaccade <number of trials> <block(0) or interleave (1)>*/

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
// Primary maintainer for this file: Po-He Tseng <ptseng@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-movie-fixicon.C $
// $Id: psycho-movie-fixicon.C 13712 2010-07-28 21:00:40Z itti $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Raster/Raster.H"
#include "Media/MPEGStream.H"
#include "Media/MediaOpts.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Video/VideoFrame.H"
#include "rutz/time.h"

#include <deque>
#include <sstream>

// ######################################################################
static int waitForFixation(const Point2D<int> fparr [], 
								           const int arraylen,
								           const int radius,         //in pixel
            							 double fixlen,            // in msec
													 nub::soft_ref<EyeTracker> et, 
													 nub::soft_ref<PsychoDisplay> d,
													 const bool do_drift_correction = false)
{
	Point2D<int> ip;				
	double dist [arraylen], sdist;
	double tt=0, adur=0, ax=0, ay=0; 
	Timer timer, timer2;
	fixlen = fixlen/1000;
	int status = 1;

	// clean up queue for key response
	while(d->checkForKey() != -1);

	// wait for fixation
	timer2.reset();
	while(timer.getSecs() < fixlen) {
		ip = et->getFixationPos();

		// check distance between fixation point to all possible targets
		for (int i=0; i<arraylen; i++)
			dist[i] = sqrt(pow(ip.i-fparr[i].i, 2) + pow(ip.j-fparr[i].j, 2));

		// get the closest target
		if (arraylen == 1)
			sdist = dist[0];
		else
			sdist = std::min(dist[0], dist[1]);

		// inside of tolerance?
		if (sdist > radius && (ip.i!=-1 && ip.j!=-1)){
			timer.reset();
		}else{
			tt = timer.getSecs()-tt;			
			adur += tt;
			ax += tt * ip.i;
		 	ay += tt * ip.j;	
		}

		//LINFO("(%i, %i) - (%i, %i) dist: %f, %f", fparr[0].i, fparr[0].j, ip.i, ip.j, sdist, timer.getSecs());

		// time out
		if (timer2.getSecs() > 5 || d->checkForKey() > 0){
			d->pushEvent("bad trial - time out / no response.");
			status = -1;
			break;
		}
	}				
	
	// do drift correction
	if (status == 1 && do_drift_correction == true) {
		// when there's only 1 target on screen			
		et->manualDriftCorrection(Point2D<double>(ax/adur, ay/adur), 
										          Point2D<double>(fparr[0].i, fparr[0].j));		
		LINFO("drift correction: (%i, %i) (%i, %i)", (int)(ax/adur), (int)(ay/adur), fparr[0].i, fparr[0].j);
	}

	return status;
}

// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Pro-Saccade");

  // Instantiate our various ModelComponents:
  nub::soft_ref<InputMPEGStream> mp
    (new InputMPEGStream(manager, "Input MPEG Stream", "InputMPEGStream"));
  manager.addSubComponent(mp);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  manager.setOptionValString(&OPT_InputMPEGStreamPreload, "true");
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "EL");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<# of trials> <block(0) or interleave(1)>", 2, 2) == false)
    return(1);

	// construct an array to indicate which trial is this:
  // 1: prosaccade, left
	// 2: prosaccade, right
	// 3: antisaccade, left
	// 4: antisaccade, right
  int ntrial = fromStr<int>(manager.getExtraArg(0).c_str());
	int interleave = fromStr<int>(manager.getExtraArg(1).c_str());
	
	LINFO("Total Trials: %i", ntrial);
	int trials[ntrial];
	int trialindex[ntrial];
	if(interleave==1){
		// interleave design
		for (int i=0; i< ntrial; i++)
			trials[i] = 1 + i % 4;
	  // randomize stimulus presentation order:
	  for (int i = 0; i < ntrial; i ++) 
			trialindex[i] = i;
	  randShuffle(trialindex, ntrial);
	}else{
		// block design
		int ntrial4 = ntrial/4;
		int tmptrials[4][ntrial4];
		int tmpidx[ntrial4];
		int blockidx[4] = {0, 1, 0, 1};
		for (int i=0; i<4; i++){
			for (int j=0; j<ntrial4; j++){
				if (i%2==0)
					tmptrials[i][j] = 1 + j%2; //prosaccades
				else
					tmptrials[i][j] = 3 + j%2; //antisaccades				
			}
		}
		for (int i=0; i<ntrial4; i++)
			tmpidx[i] = i;

		// block order (as long as the first 2 blocks are not the same)
		do {
			randShuffle(blockidx, 4);
		} while(blockidx[0] == blockidx[1]);

		// assign trial
		int c = 0;
		for (int i=0; i<4; i++){
			randShuffle(tmpidx, ntrial4);	
			for (int j=0; j<ntrial4; j++){
				trials[c] = tmptrials[blockidx[i]][tmpidx[j]];
				trialindex[c] = c;
				c++;
			}
		}			
	}

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // EyeLink opens the screen for us, so make sure SDLdisplay is slave:
  if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0)
    d->setModelParamVal("SDLslaveMode", true);

  // let's get all our ModelComponent instances started:
  manager.start();

  // vision test (generate random letters again if pressing 'r')
	LINFO("*****************************************************************************");
	LINFO("visual acuity test: [r] to regenerate another string; other keys to continue.");
	int key;
	do {
		d->displayRandomText(6, 8);
  	key = d->waitForKey(true);
	} while(key == 114 || key == 101 || key == 116 || key == 102);

	// let's do an eye tracker calibration:
	d->pushEventBegin("Calibration");
	et->setBackgroundColor(d);
	et->calibrate(d);
	d->pushEventEnd("Calibration");

	LINFO("Press any key to start......");
  d->displayText("Press any key to start......", true, 0, 10);
  d->waitForKey(true);

	// main loop
	const int x = d->getWidth()/2;
	const int y = d->getHeight()/2;
	const int dist = x;
	int trialtype = 0;
	int tx = x, txt = x;
	const PixRGB<byte> green = PixRGB<byte>(0,255,0);
	const PixRGB<byte> yellow = PixRGB<byte>(255,255,0);
	///Point2D<int> ip = Point2D<int>(0,y);;
	Point2D<int> targetpoint = Point2D<int>(0,y);
	///int status;
	Timer timer;
	const int fixation_tolerance = 135;
	int displayTask = 1;

	for (int i=0; i < ntrial; i++) {
		trialtype = trials[trialindex[i]];

		// display task instruction at the center of the screen
		d->clearScreen();

		if (i%10 == 0){
			et->recalibrate(d,13);
			d->clearScreen();
		}

		double delay = 1000000;
		if (trialtype == 1 || trialtype == 2){

			if (displayTask == 1){
				d->displayText("Same", true, 0, 10);
				if (interleave == 1)
					usleep(1000000);
				else
					d->waitForKey();
			}

			// start eye tracker
			et->track(true);
			usleep(500000);

			// involuntary, green fixation
			d->clearScreen();
			if (trialtype == 1)
				d->pushEvent("====== Trial Type: 1 (leftward, pro-saccade) ======");
			else
				d->pushEvent("====== Trial Type: 2 (rightward, pro-saccade) ======");

			// display cue, wait 0.5-1.5 seconds.....
			d->displayFixation(x+dist/2, y, false);
			d->displayFixation(x-dist/2, y, false);
			d->displayFilledCircle(x, y, 5, green);
			d->pushEvent("Display cue (green)");
			usleep(delay);

			// display target 
			tx = x + dist * (trialtype-1.5);
			timer.reset();
			//d->displayFilledCircleBlink(tx, y, 3, green, 3, 1);
			//d->displayFilledCircle(x, y, 5, d->getGrey(), false);
			d->displayFilledCircle(tx, y, 5, green);
			d->pushEvent("Display target (green)");

			// get response
			targetpoint.i = tx;
			/*status = */waitForFixation(&targetpoint, 1, fixation_tolerance, 100, et, d);	
			double tt = timer.getSecs();

			LINFO("(pro-saccade) response time: %f", tt);
			d->pushEvent(sformat("pro-saccade, response time: %f", tt));

			d->displayCircle(targetpoint.i, targetpoint.j, 60, yellow);
			usleep(1000000);

			// display information
			d->clearScreen();

		} else if (trialtype == 3 || trialtype == 4){

			if (displayTask == 1){
				d->displayText("Opposite", true, 0, 10);
				if (interleave == 1)
					usleep(1000000);
				else
					d->waitForKey();
			}

			// start eye tracker
			et->track(true);
			usleep(500000);

			// involuntary, green fixation
			d->clearScreen();
			if (trialtype == 3)
				d->pushEvent("====== Trial Type: 3 (leftward, anti-saccade) ======");
			else
				d->pushEvent("====== Trial Type: 4 (rightward, anti-saccade) ======");

			// display cue, wait 0.5-1.5 seconds.....
			d->displayFixation(x+dist/2, y, false);
			d->displayFixation(x-dist/2, y, false);
			d->displayFilledCircle(x, y, 5, green);
			d->pushEvent("Display cue (green)");
			usleep(delay);

			// display target
			tx = x + dist * (-1*(trialtype-3.5));
			txt = x + dist * (trialtype-3.5);
			timer.reset();
			//d->displayFilledCircleBlink(tx, y, 3, green, 3, 1);
			//d->displayFilledCircle(x, y, 5, d->getGrey(), false);
			d->displayFilledCircle(tx, y, 5, green);
			d->pushEvent("Display target (green)");

			// get response
			targetpoint.i = txt;
			/*status = */waitForFixation(&targetpoint, 1, fixation_tolerance, 100, et, d);	
			double tt = timer.getSecs();

			LINFO("(anti-saccade) response time: %f", tt);
			d->pushEvent(sformat("anti-saccade, response time: %f", tt));

			d->displayCircle(targetpoint.i, targetpoint.j, 60, yellow);
			usleep(1000000);

			// display information
			d->clearScreen();
		}else{
			//bug, die!!
			d->displayColorDotFixation(x, y, PixRGB<byte>(0,0,255));
			return(1);
		} 

    // stop the eye tracker:
    usleep(50000);
    et->track(false);

		// display percentage completed
		double completion = 100*((double)i+1)/double(ntrial);
		//LINFO("completion %f, %f, %f", completion, (double)i+1, double(ntrial));
		if ((int)completion % 5 == 0 && completion-(int)completion==0){
			d->clearScreen();
			d->displayText(sformat("%i %% completed", (int)completion), true, 0, 10);
			usleep(1000000);
		}

		if (interleave == 0)
			displayTask = 0;

		// take a break for a quarter of trials
		if ((int)completion%25 == 0 && i<ntrial-1 && completion-(int)completion==0) {
  		d->clearScreen();
  		d->displayText("Please Take a Break", true, 0, 10);
			LINFO("Break time.  Press [Space] to continue, or [ESC] to terminate the experiment.");
  		d->waitForKey(true);
  		d->displayText("Calibration", true, 0, 10);
      d->pushEventBegin("Calibration");
      et->calibrate(d);
      d->pushEventEnd("Calibration");
			displayTask = 1;
		}
	}

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!", true, 0, 10);
  d->waitForKey(true);

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  // simple wrapper around submain() to catch exceptions (because we
  // want to allow PsychoDisplay to shut down cleanly; otherwise if we
  // abort while SDL is in fullscreen mode, the X server won't return
  // to its original resolution)
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

  return 1;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
