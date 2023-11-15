/*!@file AppPsycho/psycho-movie.C Psychophysics display of movies */

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

#include "SDL/SDL_rotozoom.h"

#include <deque>
#include <sstream>

/*
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
*/

// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Oculomotor Capture");

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
  if (manager.parseCommandLine(argc, argv, "<# of trials>", 1, 1) == false)
    return(1);

	// construct an array to indicate target location
  int ntrial = fromStr<int>(manager.getExtraArg(0).c_str());
	LINFO("Total Trials: %i", ntrial);
	int trials[ntrial];
	for (int i=0; i< ntrial; i++)
		trials[i] = i % 6;

  // randomize stimulus presentation order:
	int trialindex[ntrial];
  for (int i = 0; i < ntrial; i ++) 
		trialindex[i] = i;
  randShuffle(trialindex, ntrial);

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

	// preparation for the main loop
	const int x = d->getWidth()/2;
	const int y = d->getHeight()/2;
	const int bradius = x/2.5; // radius for displaying 6 possible target locations
	const int sradius = 66;  // radius for target/distractor circle
	const int isradius = sradius - 8;

	//const PixRGB<byte> tcolor = PixRGB<byte>(255,0,0); // target color (red)
	const PixRGB<byte> tcolor = PixRGB<byte>(0,255,0); // target color (green)
	const PixRGB<byte> dcolor = PixRGB<byte>(255,0,0); // distractor color (red)
	const PixRGB<byte> black  = PixRGB<byte>(255,255,255);   // black color (red)
	const PixRGB<byte> bcolor = d->getGrey();          // background color

	// calulate circle locations
	// clockwise: 0, right; 1, lower-right; 2, lower-left; 3, left; 4, upper-left; 5, upper-right
	int tcirpos[6][2];
	int dcirpos[6][2];
	const double PI = 3.141592;
	for (int i=0; i<6; i++) {
		// potential target locations
		tcirpos[i][0] = x + bradius * cos(i*PI/3);	
		tcirpos[i][1] = y + bradius * sin(i*PI/3);

		// potential distractor locations
		dcirpos[i][0] = x + bradius * cos(i*PI/3 + PI/6);	
		dcirpos[i][1] = y + bradius * sin(i*PI/3 + PI/6);	
	}

	// create arrow for pointing direction
	const int arrowsize = 16;
	SDL_Surface *arrowsurf = SDL_CreateRGBSurface(SDL_SWSURFACE, arrowsize, arrowsize, 24, 0, 0, 0, 0);
	SDL_Rect arect;
	arect.x = 0; arect.y = 0; arect.w = arrowsize; arect.h = arrowsize;
	SDL_FillRect(arrowsurf, &arect, d->getUint32color(bcolor));
	arect.x = 1; arect.y = 1; arect.w = arrowsize-1; arect.h = (int)arrowsize/5;
	SDL_FillRect(arrowsurf, &arect, d->getUint32color(tcolor));
	arect.w = (int)arrowsize/5; arect.h = arrowsize-1;
	SDL_FillRect(arrowsurf, &arect, d->getUint32color(tcolor));

	// create some T and non-T for discrimination
	const int sizeT = 10;
	SDL_Surface *isT = SDL_CreateRGBSurface(SDL_SWSURFACE, sizeT, sizeT, 24, 0, 0, 0, 0);
	SDL_Surface *noT = SDL_CreateRGBSurface(SDL_SWSURFACE, sizeT, sizeT, 24, 0, 0, 0, 0);
	arect.x = 0; arect.y = 0; arect.w = sizeT; arect.h = sizeT; // background
	SDL_FillRect(isT, &arect, d->getUint32color(bcolor));
	SDL_FillRect(noT, &arect, d->getUint32color(bcolor));
	arect.x = sizeT/2; arect.y = 0; arect.w = 1; arect.h = sizeT; //vertical bar
	SDL_FillRect(isT, &arect, d->getUint32color(tcolor));
	SDL_FillRect(noT, &arect, d->getUint32color(tcolor));	
	arect.x = 0; arect.y = 0; arect.w = sizeT; arect.h = 1; //horizontal bar, T
	SDL_FillRect(isT, &arect, d->getUint32color(tcolor));
	arect.x = 0; arect.y = 2; arect.w = sizeT; arect.h = 1; //horizontal bar, non-T
	SDL_FillRect(noT, &arect, d->getUint32color(tcolor));

	SDL_Surface *isTd = SDL_CreateRGBSurface(SDL_SWSURFACE, sizeT, sizeT, 24, 0, 0, 0, 0);
	SDL_Surface *noTd = SDL_CreateRGBSurface(SDL_SWSURFACE, sizeT, sizeT, 24, 0, 0, 0, 0);
	arect.x = 0; arect.y = 0; arect.w = sizeT; arect.h = sizeT; // background
	SDL_FillRect(isTd, &arect, d->getUint32color(bcolor));
	SDL_FillRect(noTd, &arect, d->getUint32color(bcolor));
	arect.x = sizeT/2; arect.y = 0; arect.w = 1; arect.h = sizeT; //vertical bar
	SDL_FillRect(isTd, &arect, d->getUint32color(dcolor));
	SDL_FillRect(noTd, &arect, d->getUint32color(dcolor));	
	arect.x = 0; arect.y = 0; arect.w = sizeT; arect.h = 1; //horizontal bar, T
	SDL_FillRect(isTd, &arect, d->getUint32color(dcolor));
	arect.x = 0; arect.y = 2; arect.w = sizeT; arect.h = 1; //horizontal bar, non-T
	SDL_FillRect(noTd, &arect, d->getUint32color(dcolor));

  // initialze T and Not-T index that will be randomized later
  int tntidx[6];
  for(int i=0; i< 6; i++)
    tntidx[i] = i;

	// main loop
	double delay;
	int j, iscontrol, thisTarget, thisDistractor, thisSOA;
	int tnt[6], correctArray[ntrial];
	SDL_Surface *surf = SDL_CreateRGBSurface(SDL_SWSURFACE, d->getWidth(), d->getHeight(), 24, 0, 0, 0, 0);
	SDL_Rect rect, rr; rect.x = 0; rect.y = 0; rect.w = d->getWidth(); rect.h = d->getHeight();
	Timer timer;
	double rt, avg;
	avg = 0; 

	for (int i=0; i < ntrial; i++) {
		thisTarget = trials[trialindex[i]];
		thisSOA = rand()%150;
		thisDistractor = rand()%6;
		if (rand()%10 < 1){
			iscontrol = 1;
		} else {
			iscontrol = 0;
		}

		delay = 1000000; // + randomUpToIncluding(1000000);
		SDL_FillRect(surf, &rect, d->getUint32color(d->getGrey())); 

		// clean screen and do drift correction
		et->recalibrate(d,13);
		d->clearScreen();
	
    // randomize T and Not-T
    randShuffle(tntidx, 6);
    for (j=0; j<6; j++)
      tnt[tntidx[j]] = j % 2; //0, non-T; 1, T

		// track eye
		d->displayFilledCircle(x, y, 3, black, true);
		usleep(900000);
		et->track(true);
		usleep(100000);

		// display all possible target locations
		for (j=0; j<6; j++){
  		filledCircleRGBA(surf, tcirpos[j][0], tcirpos[j][1], sradius, tcolor.red(), tcolor.green(), tcolor.blue(), 0XFF);
  		filledCircleRGBA(surf, tcirpos[j][0], tcirpos[j][1], isradius, bcolor.red(), bcolor.green(), bcolor.blue(), 0XFF);
		
			rr.x = tcirpos[j][0]-isT->w/2; rr.y = tcirpos[j][1]-isT->h/2;
			if (tnt[j] == 0) {
				SDL_BlitSurface(noT, NULL, surf, &rr);
			} else {
				SDL_BlitSurface(isT, NULL, surf, &rr);
			}
		}

		// is it an control trial?
		if (iscontrol == 1) {
  		filledCircleRGBA(surf, dcirpos[thisDistractor][0], dcirpos[thisDistractor][1], sradius, tcolor.red(), tcolor.green(), tcolor.blue(), 0XFF);
  		filledCircleRGBA(surf, dcirpos[thisDistractor][0], dcirpos[thisDistractor][1], isradius, bcolor.red(), bcolor.green(), bcolor.blue(), 0XFF);
		}

		// display array
  	filledCircleRGBA(surf, x, y, 3, black.red(), black.green(), black.blue(), 0XFF);
		d->displaySurface(surf, -2, true);
		d->pushEvent("Array UP");

		// waiting before target comes up
		usleep(delay);

		// show direction
		LINFO("thisTarget: %i, isT: %i", thisTarget, tnt[thisTarget]);
		
		// show 
		for (j=0; j<6; j++)
			if (j != thisTarget){
  			filledCircleRGBA(surf, tcirpos[j][0], tcirpos[j][1], sradius, dcolor.red(), dcolor.green(), dcolor.blue(), 0XFF);
  			filledCircleRGBA(surf, tcirpos[j][0], tcirpos[j][1], isradius, bcolor.red(), bcolor.green(), bcolor.blue(), 0XFF);

				rr.x = tcirpos[j][0]-isT->w/2; rr.y = tcirpos[j][1]-isT->h/2;
				if (tnt[j] == 0) {
					SDL_BlitSurface(noTd, NULL, surf, &rr);
				} else {
					SDL_BlitSurface(isTd, NULL, surf, &rr);
				}
			}

		timer.reset();
		if (thisSOA < 34){
  		filledCircleRGBA(surf, dcirpos[thisDistractor][0], dcirpos[thisDistractor][1], sradius, dcolor.red(), dcolor.green(), dcolor.blue(), 0XFF);
  		filledCircleRGBA(surf, dcirpos[thisDistractor][0], dcirpos[thisDistractor][1], isradius, bcolor.red(), bcolor.green(), bcolor.blue(), 0XFF);
			d->displaySurface(surf, -2, true);
			d->pushEvent(sformat("Target direction shown: %d", thisTarget));
			if (iscontrol == 1)
				d->pushEvent("Distractor shown: control condition");
			else
				d->pushEvent(sformat("Distractor shown: %f (SOA = %d ms)", thisDistractor+0.5, thisSOA));
		
		} else {

			// display instruction direction first
			d->displaySurface(surf, -2, true);
			d->pushEvent(sformat("Target direction shown: %d", thisTarget));

			// wait
			usleep(thisSOA*1000);

			// display distractor
  		filledCircleRGBA(surf, dcirpos[thisDistractor][0], dcirpos[thisDistractor][1], sradius, dcolor.red(), dcolor.green(), dcolor.blue(), 0XFF);
  		filledCircleRGBA(surf, dcirpos[thisDistractor][0], dcirpos[thisDistractor][1], isradius, bcolor.red(), bcolor.green(), bcolor.blue(), 0XFF);
			d->displaySurface(surf, -2, true);
			d->pushEvent(sformat("Distractor shown: %f (SOA = %d ms)", thisDistractor+0.5, thisSOA));
		
		}	

		// wait for response	
		//LINFO("Press 4 for T, 6 for Not-T, or Space Bar to skip.");
		do {
			key = d->waitForKey();
		} while (key != 52 && key != 54 && key != 32);
		rt = timer.getMilliSecs();
		int kk = 0;
		if (key==52)
			kk = 1;
		else if (key==32)
			kk = -1; // space bar

		d->pushEvent(sformat("Responded %d (%f ms), correct %d", kk, rt, kk==tnt[thisTarget]));
		LINFO("Responded %d (%f ms), correct %d", kk, rt, kk==tnt[thisTarget]);
		avg = onlineMean(avg, rt, i+1);

		d->clearScreen();

		usleep(50000);
		et->track(false);
	
		correctArray[i] = (kk==tnt[thisTarget]);

		// display percentage completed
		double completion = 100*((double)i+1)/double(ntrial);
		if ((int)completion % 5 == 0 && completion - (int)completion == 0){
		
			d->clearScreen();
			d->displayText(sformat("%i %% completed", (int)completion), true, 0, 10);
			usleep(1000000);
		}

		// take a break for a quarter of trials
		if ((int)completion%50 == 0 && i<ntrial-1 && completion-(int)completion==0) {
  		d->clearScreen();
  		d->displayText("Please Take a Break", true, 0, 10);

			// performance information
			int cc = 0;
			for (j=0; j<=i; j++)
				if (correctArray[j]==1)
					cc++;
			LINFO("Average RT: %f ms, Correctness: %f%%", avg, 100*(double)cc/((double)i+1));

			LINFO("Break time.  Press [Space] to continue, or [ESC] to terminate the experiment.");
  		d->waitForKey(true);
  		d->displayText("Calibration", true, 0, 10);
      d->pushEventBegin("Calibration");
      et->calibrate(d);
      d->pushEventEnd("Calibration");
		}
	}

	SDL_FreeSurface(arrowsurf);
	SDL_FreeSurface(surf);

	// performance information
	int cc = 0;
	for (j=0; j<ntrial; j++)
		if (correctArray[j]==1)
			cc++;
	LINFO("Average RT: %f ms, Correctness: %f%%", avg, 100*(double)cc/((double)ntrial));

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
