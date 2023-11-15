/*!@file AppPsycho/psycho-motion.C Psychophysics display of randomly moving
  clouds of dots for checking motion coherence */

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
// Primary maintainer for this file: Vidhya Navalpakkam <navalpak@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-motion.C $
// $Id: psycho-motion.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Component/ModelManager.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "GUI/GUIOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Video/RgbConversion.H" // for toVideoYUV422()


// create a circular cloud of random dots at center (x,y) and radius r
void createCloud (DOT* dots, int numDots, int x, int y, int r, int move);
// set coherence of the cloud
void setCoherence (DOT* dots, int numDots,  int numCoherent, int move);

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho motion");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  // set some display params
  manager.setOptionValString(&OPT_SDLdisplayDims, "640x480");
  d->setModelParamVal("PsychoDisplayBackgroundColor", PixRGB<byte>(0));
  d->setModelParamVal("PsychoDisplayTextColor", PixRGB<byte>(255));
  d->setModelParamVal("PsychoDisplayBlack", PixRGB<byte>(255));
  d->setModelParamVal("PsychoDisplayWhite", PixRGB<byte>(128));
  d->setModelParamVal("PsychoDisplayFixSiz", 5);
  d->setModelParamVal("PsychoDisplayFixThick", 5);
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");


  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<radius> <midCoherence> "
                               "<targetCoherence> <numDots> <life> "
                               "<waitFrames> <move> <startTrial#>",
                               1, -1)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  // let's display an ISCAN calibration grid:
  d->clearScreen();
  d->displayISCANcalib();
  d->waitForKey();

  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration(3, 3);

  d->clearScreen();

  // read the radius of the cloud and the coherence
  int radius = manager.getExtraArgAs<int>(0);
  int midCoherence = manager.getExtraArgAs<int>(1);
  int targetCoherence = manager.getExtraArgAs<int>(2);
  int numDots = manager.getExtraArgAs<int>(3);
  int life = manager.getExtraArgAs<int>(4);
  int waitNum = manager.getExtraArgAs<int>(5);
  int move = manager.getExtraArgAs<int>(6);
  int startTrial = manager.getExtraArgAs<int>(7);

  // show a fixation cross on a blank screen:
  d->clearScreen();

  initRandomNumbers();
  int numTrial = 0;
  DOT clouds[25][numDots];                     // allocate space for dots
  Image< PixRGB<byte> > bufima(d->getDims(), NO_INIT);

  // create background stimuli
  int leg = 15; int bgDots = 2 * leg - 1;
  DOT targetShape[bgDots], distractorShape[bgDots];
  for (int i = 0; i < bgDots; i++) {
    if (i < leg) {
      (targetShape[i]).x = 0; (targetShape[i]).y = -i;
      (distractorShape[i]).x = 0; (distractorShape[i]).y = -i;
    }
    else {
     (targetShape[i]).x = leg-i; (targetShape[i]).y = -leg;
      (distractorShape[i]).x = i-leg; (distractorShape[i]).y = 0;
    }
  }
  DOT bg[25][bgDots];                         // allot space for background

  while (numTrial < 30)
  {
    d->createYUVoverlay(SDL_YV12_OVERLAY);
    char reco[10]; sprintf (reco, "reco-%d", startTrial + numTrial);
    numTrial ++;
    FILE* f = fopen (reco, "w");
    // ready to go whenever the user is ready:
    d->displayText("hit any key when ready");
    d->waitForKey();
    d->waitNextRequestedVsync(false, true);
    // create stimulus on the fly
    int targetCloud = -1;
    // decide position of LESS, MID, HIGH and target
    int index[25];
    for (int i = 0; i < 25; i++)
      index[i] = i;
    randShuffle (index, 25);
    // draw target and distractor clouds
    int cenx[25], ceny[25];
    for (int i = 0; i < 25; i++){
      int idx = index[i];
      int col = idx % 5, row = idx / 5;
      cenx[idx] = col * 128 + 64 + (int) (randomDouble()*31) - 15;
      ceny[idx] = row * 96 + 48 + (int) (randomDouble()*25) - 12 ;
      DOT* dots = clouds[idx];                 // dots of this cloud
      createCloud (dots, numDots, cenx[idx], ceny[idx], radius, move);
      if (i == 24) {                                // TARGET type
        setCoherence (dots, numDots,
                      (int) (targetCoherence*numDots/100), move);
        targetCloud = idx;
        LINFO ("target location: (%d,%d)", row, col);
        fprintf (f, "%d %d %d %d target\n", row, col, cenx[idx],ceny[idx]);
        // generate background stimulus
        for (int j = 0; j < bgDots; j++){
          DOT* dot_bg = bg[idx] + j;
          dot_bg->x = (targetShape[j]).x + cenx[idx] + leg/2;
          dot_bg->y = (targetShape[j]).y + ceny[idx] + leg/2;
        }
      }
      else {                                          // DISTRACTOR type
        for (int j = 0; j < bgDots; j++){
          DOT* dot_bg = bg[idx] + j;
          dot_bg->x = (distractorShape[j]).x + cenx[idx] - leg/2;
          dot_bg->y = (distractorShape[j]).y + ceny[idx] + leg/2;
        }
        if (i < 8){                                   // LESS type
          setCoherence (dots, numDots, 0, move);
          fprintf (f, "%d %d %d %d less\n",
                   row, col, cenx[idx], ceny[idx]);
        }
        else if (i < 16){                             // MID type
          setCoherence (dots, numDots,
                        (int) (midCoherence*numDots/100), move);
          fprintf (f, "%d %d %d %d mid\n", row, col, cenx[idx], ceny[idx]);
        }
        else if (i < 24){                             // HIGH type
          setCoherence (dots, numDots, numDots, move);
          fprintf (f, "%d %d %d %d high\n",
                   row, col, cenx[idx], ceny[idx]);
        }
      }
    }
    fclose (f);                                       // close the reco file

    // start the eye tracker:
    et->track(true);

    // blink the fixation:
    d->displayFixationBlink();
    // generate the stimulus on the fly and display it until key press
    int time = 0;
    while (d->checkForKey() == -1){
      time ++;
      // at each time instant, update position of the dots in each cloud
      for (int i = 0; i < 25; i++){
        int idx = index[i];
        /*
          The following probabilistic renewal is according to Britten
          et.  al(1992). But it causes more flicker in the random case
          and none in the coherent case.

          if (i < 8) renewal = 0.0f;
          else if (i < 16) renewal = midCoherence/100.0f;
          else if (i < 24) renewal = 1.0f;
          else renewal = targetCoherence/100.0f;
        */
        for (int j = 0; j < numDots; j++){         // update each cloud
          DOT *dot = &(clouds[idx][j]);
          int x1 = dot->x - cenx[idx];
          int y1 = dot->y - ceny[idx];                  // dist wrt center
          /* similar to Britten et. al (1992)
             if (randomDouble() > renewal) {
             int size = 2 * radius + 1;
             x1 = (int) (randomDouble()*size) - radius;
             y1 = (int) (randomDouble()*size) - radius;
             }
          */
          if (dot->coherent == 1) y1 += dot->dy;
          else {
            // move the random dot in a random direction
            if (dot->age > life){
              // choose a new random direction
              dot->age = 0;
              double rand = randomDouble();
              if (rand <= 0.33) dot->dx = move;
              else if (rand <= 0.66) dot->dx = -move;
              else dot->dx = 0;
              rand = randomDouble();
              if (rand <= 0.33) dot->dy = move;
              else if (rand <= 0.66) dot->dy = -move;
              else dot->dy = 0;
              if (dot->dx == 0 && dot->dy == 0){
                if (randomDouble() <= 0.5) dot->dy = move;
                else dot->dy = -move;
              }
            }
            x1 += dot->dx;
            y1 += dot->dy;
            dot->age += 1;
          }
          // reappearance of a dot that crosses the boundary
          if (x1 < -radius) x1 = radius;
          else if (x1 > radius) x1 = -radius;
          if (y1 < -radius) y1 = radius;
          else if (y1 > radius) y1 = -radius;
          // absolute position of the dot
          dot->x = x1 + cenx[idx];
          dot->y = y1 + ceny[idx];
        }
      }

      // draw the updated clouds
      bufima.clear();

      for (int i = 0; i < 25; i++)
        for (int j = 0; j < numDots; j++){
          int x = (clouds[i][j]).x;
          int y = (clouds[i][j]).y;
          drawDisk(bufima, Point2D<int>(x, y), 2, PixRGB<byte>(120));
        }
      for (int i = 0; i < 25; i++)
        for (int j = 0; j < bgDots; j++){
          int x = (bg[i][j]).x;
          int y = (bg[i][j]).y;
          bufima.setVal(x, y, PixRGB<byte>(75));
        }


      SDL_Overlay *ovl = d->lockYUVoverlay();
      toVideoYUV422(bufima,
                    ovl->pixels[0], ovl->pixels[1], ovl->pixels[2]);
      d->unlockYUVoverlay();
      d->displayYUVoverlay(time, SDLdisplay::NEXT_VSYNC);

      d->waitFrames(waitNum);
    }
    // stop the eye tracker:
    usleep(50000);
    et->track(false);

    // no cheating: flash a grid of random numbers and check response
    int correctResponse = d->displayNumbers (targetCloud/5,
                                             targetCloud%5, true);
    d->pushEvent(std::string("===== Showing noCheat ====="));

    // flash the image for 99ms:
    for (int j = 0; j < 10; j ++) d->waitNextRequestedVsync();

    // wait for response:
    d->displayText("Enter the number at the target location");
    c = d->waitForKey();
    int c2 = d->waitForKey();
    // check if the response is correct
    int observedResponse = 10*(c-48) + c2-48;
    LINFO (" subject entered %d and correct response is %d",
           observedResponse, correctResponse);
    if (observedResponse == correctResponse)
      {
        d->displayText("Correct!");
        d->pushEvent(std::string("===== Correct ====="));
      }
    else
      {
        d->displayText("Wrong! ");
        d->pushEvent(std::string("===== Wrong ====="));
      }
    // maintain display
    for (int j = 0; j < 30; j ++) d->waitNextRequestedVsync();
    d->destroyYUVoverlay();
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
// create a circular cloud of random dots at center (x,y) and radius r
void createCloud (DOT* dots, int numDots, int cx, int cy, int r, int move)
{
  int i = 0;
  int size = 2 * r + 1;
  while (i < numDots){
    // generate a dot
    int x = (int) (randomDouble()*size) - r;
    int y = (int) (randomDouble()*size) - r;
    dots[i].x = cx + x;                  // set x coordinate
    dots[i].y = cy + y;                  // set y coordinate
    dots[i].coherent = 0;                // set coherent
    dots[i].age = (int) (randomDouble() * 5);
    double rand = randomDouble();
    if (rand <= 0.33) dots[i].dx = move;
    else if (rand <= 0.66) dots[i].dx = -move;
    else dots[i].dx = 0;
    rand = randomDouble();
    if (rand <= 0.33) dots[i].dy = move;
    else if (rand <= 0.66) dots[i].dy = -move;
    else dots[i].dy = 0;
    if (dots[i].dx == 0 && dots[i].dy == 0){
      if (randomDouble() <= 0.5) dots[i].dy = move;
      else dots[i].dy = -move;
    }
    i++;
  }
}
// ######################################################################
// set coherence of the cloud
void setCoherence (DOT* dots, int numDots, int numCoherent, int move)
{
  // initialize an array of dot indices
  int index[numDots];
  for (int i = 0; i < numDots; i++)
    index[i] = i;

  // randomize index
  randShuffle (index, numDots);

  // set coherence of the dots
  for (int i = 0; i < numCoherent; i++){
    dots[index[i]].coherent = 1;
    dots[index[i]].dx = 0;
    dots[index[i]].dy = move;
  }
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
