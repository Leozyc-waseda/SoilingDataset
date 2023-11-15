/*!@file AppPsycho/psycho-obfba2.C Psychophysics display */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-obfba2.C $
// $Id: psycho-obfba2.C 13712 2010-07-28 21:00:40Z itti $
//

#include "Component/ModelManager.H"
#include "Image/CutPaste.H"  // for inplacePaste()
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/Kernels.H"   // for gaborFilter()
#include "Psycho/PsychoDisplay.H"
#include "GUI/GUIOpts.H"
#include "Psycho/Staircase.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/Timer.H"

#include <deque>
#include <fstream>

#define GABORSD   15.0F
#define GABORPER  15.0F

// number of frames for scanner setup:
#define NFSETUP 12*60

// number of wait frames during which instruction message is displayed
#define NFINIT 36

// number of frames per stimulus interval:
#define NFSTIM 37

// number of frames per inter-period:
#define NFIPI 15

 // number of blank frames between two trials
#define NFISI 55

// number of trials
#define NTRIAL 12

/* for jianwei first run (sunday):
   init=29 stim=17 ipi=7 isi=26 trial=25 (total 1675)*/

/* Experiment file format is:
   # message                      message
   ## message                     message with a pause before
   ABC CDE
*/

//! Set this to true if you want to block until responses are given
const bool blocking = false;

//! Set this to true if you want to add drop shadows
bool shadow = true;

//! Types of tasks
enum TaskType { None = 0, Orientation = 1, Drift = 2, Blank = 3 };

//! A summary struct to handle task settings
struct PsychoTask {
  nub::soft_ref<Staircase> s;
  TaskType tt;
  float ampl;
  double theta;
  double drift;
};

//! Parse task definitions given at command line
void parseTaskDefinition(const std::string& arg, PsychoTask& p,
                         const std::string name)
{
  if (arg.length() != 3) LFATAL("Task definition should have 3 chars");
  p.s->stop();
  p.s->setModelParamVal("FileName", name);
  switch(arg[0])
    {
    case 'N':
      p.tt = None;
      p.ampl = 127.0f;
      p.s->setModelParamVal("InitialValue", 0.0);
      p.s->setModelParamVal("DeltaValue", 0.0);
      p.s->setModelParamVal("MinValue", 0.0);
      p.s->setModelParamVal("MaxValue", 0.0);
      break;
    case 'O':
      p.tt = Orientation;
      p.ampl = 127.0f;
      p.s->setModelParamVal("InitialValue", 4.0);
      p.s->setModelParamVal("DeltaValue", 0.5);
      p.s->setModelParamVal("MinValue", 4.0);
      p.s->setModelParamVal("MaxValue", 4.0);
      break;
    case 'D':
      p.tt = Drift;
      p.ampl = 127.0f;
      p.s->setModelParamVal("InitialValue", 8.0);
      p.s->setModelParamVal("DeltaValue", 2.0);
      p.s->setModelParamVal("MinValue", 8.0);
      p.s->setModelParamVal("MaxValue", 8.0);
      break;
    case 'B':
      p.tt = Blank;
      p.ampl = 0.0f;
      p.s->setModelParamVal("InitialValue", 0.0);
      p.s->setModelParamVal("DeltaValue", 0.0);
      p.s->setModelParamVal("MinValue", 0.0);
      p.s->setModelParamVal("MaxValue", 0.0);
      break;
    default:
      LFATAL("Incorrect task '%c'", arg[0]);
    }

  switch(arg[1])
    {
    case 'V': p.theta = 0.0; break;
    case 'H': p.theta = 90.0; break;
    default: LFATAL("Incorrect orientation definition '%c'", arg[1]);
    }

  switch(arg[2])
    {
    case 'F': p.drift = 28.0; break;
    case 'S': p.drift = 16.0; break;
    default: LFATAL("Incorrect drift definition '%c'", arg[1]);
    }
  p.s->start();
}


// ######################################################################
//! Psychophysics display
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho OBFBA");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  manager.setOptionValString(&OPT_SDLdisplayDims, "640x480");
  manager.setOptionValString(&OPT_SDLdisplayRefreshUsec, "16666.66666"); //60Hz

  PsychoTask p[2];
  p[0].s.reset(new Staircase(manager, "obfbaL", "obfbaL"));
  manager.addSubComponent(p[0].s);
  p[1].s.reset(new Staircase(manager, "obfbaR", "obfbaR"));
  manager.addSubComponent(p[1].s);

  Image< PixRGB<byte> > background = Raster::ReadRGB("b640.ppm");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<taskfile.txt>", 1, 1) == false)
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // make sure we catch exceptions and revert to normal display in case of bug:
  try {
    // open the taskfile:
    std::string fname = manager.getExtraArg(0);
    std::ifstream ifs(fname.c_str());
    if (ifs.is_open() == false)
      LFATAL("Cannot read %s", manager.getExtraArg(0).c_str());
    std::deque<std::string> line;
    while(1) {
      char str[1000];
      ifs.getline(str, 1000);
      if (strlen(str) == 0) break;
      line.push_back(std::string(str));
    }
    ifs.close();

    // let's build the background image:
    int w = d->getDims().w(), h = d->getDims().h();
    PixRGB<byte> black(0), grey(d->getGrey()), white(255);

    Image< PixRGB<byte> > background2 = background;
    Image< PixRGB<byte> > blk(w-80, h-120, NO_INIT);
    blk.clear(PixRGB<byte>(grey));
    inplacePaste(background, blk, Point2D<int>(40, 60));

    // draw a fixation cross in our image:
    drawCross(background, Point2D<int>(w/2, h/2), black, 10, 2);
    drawCross(background, Point2D<int>(w/2, h/2), white, 10, 1);
    drawCross(background2, Point2D<int>(w/2, h/2), black, 10, 2);
    drawCross(background2, Point2D<int>(w/2, h/2), white, 10, 1);

    // find out the size and position of the Gabor patches:
    Image<float> tmp = gaborFilter<float>(GABORSD, GABORPER, 0.0F, 0.0F);
    int pw = tmp.getWidth(), ph = tmp.getHeight();
    LINFO("Gabor patch size: %dx%d", pw, ph);
    Point2D<int> lpos((w/2 - pw) / 2, (h - ph) / 2);
    Point2D<int> rpos(w/2 + (w/2 - pw) / 2, (h - ph) / 2);
    Image< PixRGB<byte> > blank(tmp.getDims(), NO_INIT);
    blank.clear(PixRGB<byte>(grey));

    // draw grey boxes in our background image around the gabors:
    int border = 40;  // border size all round each gabor
    Image<float> mult(background2.getDims(), NO_INIT);
    mult.clear(1.0F);
    Image<float> mask(blank.getWidth() + border * 2,
                      blank.getHeight() + border * 2, NO_INIT);
    mask.clear(0.5F);
    int off = 10;
    inplacePaste(mult, mask, Point2D<int>(lpos.i-border+off, lpos.j-border+off));
    inplacePaste(mult, mask, Point2D<int>(rpos.i-border+off, rpos.j-border+off));
    background2 *= mult;
    Image< PixRGB<byte> > tmp2(blank.getWidth() + border*2,
                               blank.getHeight() + border * 2, NO_INIT);
    tmp2.clear(PixRGB<byte>(grey));
    inplacePaste(background2, tmp2, Point2D<int>(lpos.i - border, lpos.j - border));
    inplacePaste(background2, tmp2, Point2D<int>(rpos.i - border, rpos.j - border));

    // prepare a blittable image surface from the background image:
    SDL_Surface *bgimg = d->makeBlittableSurface(background);
    SDL_Surface *bg2img = d->makeBlittableSurface(background2);

    // ready to go:
    Timer tim, tim2, tim3;
    if (system("/bin/sync")) LERROR("error in sync");
    usleep(500000);

    // let's do the scanner setup:
    d->displayText("Ready");
    while(d->waitForKey() != ' ') ;  // wait for SPACE keypress
    tim.reset();
    d->clearScreen(true);
    d->waitFrames(NFSETUP);
    LINFO("Scanner warmup took %llxms", tim.get());
    int count = 0;

    std::string msg; bool do_wait = false;
    while(line.size() > 0)
      {
        std::string str = line.front(); line.pop_front();

        // is it a message?
        if (str[0] == '#')
          {
            msg = str; msg.erase(0, 1); do_wait = false;
            if (msg[0] == '#') { msg.erase(0, 1); do_wait = true; }
          }
        else
          // it's a double task definition
          {
            tim3.reset();
            // initialize our staircases:
            for (int i = 0; i < 2; i ++) {
              std::string xx(str.c_str() + i*4, 3);
              std::string pname("STAIR-");
              if (i == 0) pname += "L"; else pname += "R";
              char gogo[10]; sprintf(gogo, "%03d", count);
              pname += gogo; pname += fname;
              parseTaskDefinition(xx, p[i], pname);
            }
            count ++;

            // prepare the Gabors:
            float sd = GABORSD, per = GABORPER;
            Image< PixRGB<byte> > left, right;

            // wait for key if requested
            if (do_wait) d->waitForKey();

            // ready to go:
            LINFO("%s", msg.c_str());
            d->displayText(msg.c_str());
            d->waitFrames(NFINIT);
            d->waitNextRequestedVsync(false, true);

            // show background image and fixation cross:
            if (str.length() > 8)
              d->displaySurface(bg2img, -2);
            else
              d->displaySurface(bgimg, -2);

            // go over the trials:
            for (int trial = 0; trial < NTRIAL; trial ++)
              {
                tim.reset();
                tim2.reset();

                int nf = NFSTIM;  // number of frames per stimulus interval
                int nb1 = NFIPI;  // number of frames per inter-period
                int nb2 = NFISI; // number of blank frames between two trials
                float base = d->getGrey().luminance(); // base Gabor grey
                int rphil = randomUpToIncluding(90);
                int rphir = randomUpToIncluding(90);

                int idxL = 0, idxR = 1;
                double dl1, dl2, dr1, dr2;
                p[idxL].s->getValues(dl1, dl2);
                p[idxR].s->getValues(dr1, dr2);

                double dt1L = 0.0, dt2L = 0.0, dd1L = 0.0, dd2L = 0.0,
                  dt1R = 0.0, dt2R = 0.0, dd1R = 0.0, dd2R = 0.0;
                switch(p[idxL].tt)
                  {
                  case None:
                  case Blank:
                    dt1L = 0.0; dt2L = 0.0; dd1L = 0.0; dd2L = 0.0;
                    break;
                  case Orientation:
                    dt1L = dl1; dt2L = dl2; dd1L = 0.0; dd2L = 0.0;
                    break;
                  case Drift:
                    dt1L = 0.0; dt2L = 0.0; dd1L = dl1; dd2L = dl2;
                    break;
                  }
                switch(p[idxR].tt)
                  {
                  case None:
                  case Blank:
                    dt1R = 0.0; dt2R = 0.0; dd1R = 0.0; dd2R = 0.0;
                    break;
                  case Orientation:
                    dt1R = dr1; dt2R = dr2; dd1R = 0.0; dd2R = 0.0;
                    break;
                  case Drift:
                    dt1R = 0.0; dt2R = 0.0; dd1R = dr1; dd2R = dr2;
                    break;
                  }

                // stimulus presentation period 1:
                for (int i = 0; i < nf; i ++)
                  {
                    left =
                      gaborFilter<byte>(sd, per,
                                        i*(p[idxL].drift + dd1L) + rphil,
                                        p[idxL].theta + dt1L, base,
                                        p[idxL].ampl);
                    right =
                      gaborFilter<byte>(sd, per,
                                        i*(p[idxR].drift + dd1R) + rphir,
                                        p[idxR].theta + dt1R, base,
                                        p[idxR].ampl);
                    d->displayImagePatch(left, lpos, i, false, false);
                    d->displayImagePatch(right, rpos, i, true, true);
                  }
                long int t0 = tim.getReset();

                // inter-stimulus blank:
                d->displayImagePatch(blank, lpos, -2, false, false);
                d->displayImagePatch(blank, rpos, -2);
                d->waitFrames(nb1);
                long int t1 = tim.getReset();

                // stimulus presentation period 2:
                rphil = randomUpToIncluding(90);
                rphir = randomUpToIncluding(90);
                for (int i = 0; i < nf; i ++)
                  {
                    left =
                      gaborFilter<byte>(sd, per,
                                        i*(p[idxL].drift + dd2L) + rphil,
                                        p[idxL].theta + dt2L, base,
                                        p[idxL].ampl);
                    right =
                      gaborFilter<byte>(sd, per,
                                        i*(p[idxR].drift + dd2R) + rphir,
                                        p[idxR].theta + dt2R, base,
                                        p[idxR].ampl);
                    d->displayImagePatch(left, lpos, i, false, false);
                    d->displayImagePatch(right, rpos, i, true, true);
                  }
                long int t2 = tim.getReset();

                // inter-trial blank:
                d->displayImagePatch(blank, lpos, -2, false, false);
                d->displayImagePatch(blank, rpos, -2);
                if (blocking == false) d->waitFrames(nb2);

                // collect the response(s):
                if (p[idxL].tt != None && p[idxL].tt != Blank)
                  {
                    int c;
                    if (blocking) c = d->waitForKey();
                    else c = d->checkForKey();
                    if (c != -1) p[idxL].s->setResponse( (c == '\r') );
                    else p[idxL].s->setResponse( (randomDouble() < 0.5) ); // rnd
                  }
                else p[idxL].s->setResponse(false);

                if (p[idxR].tt != None && p[idxR].tt != Blank)
                  {
                    int c;
                    if (blocking) c = d->waitForKey();
                    else c = d->checkForKey();
                    if (c != -1) p[idxR].s->setResponse( (c == '\r') );
                    else p[idxR].s->setResponse( (randomDouble() < 0.5) ); // rnd
                  }
                else p[idxR].s->setResponse(false);

                long int t3 = tim.getReset();
                long int tt = tim2.get();
                float pe = 16.666666f;
                LINFO("Trial %d: p1=%ldms (%df) ipi=%ldms (%df) p2=%ldms (%df) "
                      "isi=%ldms (%df) tot=%ldms (%df)",
                      trial,
                      t0, int(t0/pe + 0.4999f),
                      t1, int(t1/pe + 0.4999f),
                      t2, int(t2/pe + 0.4999f),
                      t3, int(t3/pe + 0.4999f),
                      tt, int(tt/pe + 0.4999f));
              }

            // reset the staircases before the next epoch:
            for (int i = 0; i < 2; ++i)
              p[i].s->reset(MC_RECURSE);

            float pe = 16.666666f;
            long int tt = tim3.get();
            LINFO("Task took %ldms (%df)", tt, int(tt/pe + 0.4999f));
          }
      }

    d->clearScreen();
    d->displayText("Experiment complete. Thank you!");
    d->waitFrames(100);
  }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

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
