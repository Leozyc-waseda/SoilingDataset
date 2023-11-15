/*!@file CMapDemo/test-SaliencyCMapMT.C tests the multi-threaded salincy code with
        a cmap corba object
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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CMapDemo/test-SaliencyCMapMT.C $
// $Id: test-SaliencyCMapMT.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef TESTSALIENCYMT_H_DEFINED
#define TESTSALIENCYMT_H_DEFINED

#include "CMapDemo/SaliencyCMapMT.H"
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWinManaged.H"
#include "Image/Convolver.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/SaccadeControllers.H"
#include "Neuro/WTAwinner.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#define MAXFLOAT       3.40282347e+38F
#define sml 0

//! Number of frames over which average framerate is computed
#define NAVG 20

//! Factor to display the sm values as greyscale:
#define SMFAC 1.00F

#define WINSIZE 51
static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

ImageSet<float> bias(14);
ImageSet<float> newBias(14);



// ######################################################################
int main(int argc, char **argv)
{
        MYLOGVERB = LOG_INFO;

        CORBA::ORB_ptr orb = CORBA::ORB_init(argc,argv,"omniORB4");

        // instantiate a model manager (for camera input):
        ModelManager manager("SaliencyMT Tester");

        // Instantiate our various ModelComponents:
        nub::ref<FrameGrabberConfigurator>
                gbc(new FrameGrabberConfigurator(manager));
        manager.addSubComponent(gbc);

        nub::ref<SaliencyMT> smt(new SaliencyMT(manager, orb, sml));
        manager.addSubComponent(smt);


        // Set the appropriate defaults for our machine that is connected to
        // Stefan's robot head:
        manager.exportOptions(MC_RECURSE);
        manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
        manager.setOptionValString(&OPT_FrameGrabberMode, "YUV420P");
        manager.setOptionValString(&OPT_FrameGrabberDevice, "/dev/video1");
        manager.setOptionValString(&OPT_FrameGrabberChannel, "1");
        manager.setOptionValString(&OPT_FrameGrabberHue, "0");
        manager.setOptionValString(&OPT_FrameGrabberContrast, "16384");
        // manager.setOptionValString(&OPT_FrameGrabberDims, "320x240");
        manager.setOptionValString(&OPT_FrameGrabberDims, "160x120");
        manager.setOptionValString(&OPT_SaccadeControllerEyeType, "Threshfric");
        manager.setOptionValString(&OPT_SCeyeMaxIdleSecs, "1000.0");
        manager.setOptionValString(&OPT_SCeyeThreshMinOvert, "4.0");
        manager.setOptionValString(&OPT_SCeyeThreshMaxCovert, "3.0");
        manager.setOptionValString(&OPT_SCeyeThreshMinNum, "2");
        //  manager.setOptionValString(&OPT_SCeyeSpringK, "1000000.0");

        //manager.setOptionValString(&OPT_SaccadeControllerType, "Trivial");

        // Parse command-line:
        if (manager.parseCommandLine((const int)argc, (const char**)argv, "", 0, 0) == false) return(1);

        // do post-command-line configs:
        nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
        if (gb.isInvalid())
                LFATAL("You need to select a frame grabber type via the "
                                "--fg-type=XX command-line option for this program "
                                "to be useful");
        const int w = gb->getWidth(), h = gb->getHeight();

        const int foa_size = std::min(w, h) / 12;
        manager.setModelParamVal("InputFrameDims", Dims(w, h),
                        MC_RECURSE | MC_IGNORE_MISSING);
        manager.setModelParamVal("SCeyeStartAtIP", true,
                        MC_RECURSE | MC_IGNORE_MISSING);
        manager.setModelParamVal("SCeyeInitialPosition",Point2D<int>(w/2,h/2),
                        MC_RECURSE | MC_IGNORE_MISSING);
        manager.setModelParamVal("FOAradius", foa_size,
                        MC_RECURSE | MC_IGNORE_MISSING);
        manager.setModelParamVal("FoveaRadius", foa_size,
                        MC_RECURSE | MC_IGNORE_MISSING);

        // catch signals and redirect them to terminate for clean exit:
        signal(SIGHUP, terminate); signal(SIGINT, terminate);
        signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
        signal(SIGALRM, terminate);

        // setup socket:
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock == -1) PLFATAL("Cannot create server socket");

        // get prepared to grab, communicate, display, etc:
        uint frame = 0U;                  // count the frames
        uint lastframesent = 0U;          // last frame sent for processing

        uint64 avgtime = 0; int avgn = 0; // for average framerate
        float fps = 0.0F;                 // to display framerate
        uint64 p_avgtime = 0; int p_avgn = 0; // for average framerate
        float p_fps = 0.0F;                 // to display framerate
        Timer tim;                        // for computation of framerate
        Timer p_tim;                        // for computation of framerate
        Timer masterclock;                // master clock for simulations

        Image<float> sm(w >> sml, h >> sml, ZEROS); // saliency map
        Point2D<int> fixation(-1, -1);         // coordinates of eye fixation

        // image buffer for display:
        Image<PixRGB<byte> > disp(w * 2, h + 20, ZEROS);
        disp += PixRGB<byte>(128);
        XWinManaged xwin(disp.getDims(), -1, -1, "RCBot");
        XWinManaged xwin2(Dims(w,h), -1, -1, "RCBot");
        XWinManaged xwin3(Dims(256,256), -1, -1, "RCBot");

        ///  int ovlyoff = (w * 2) * ((dh - disp.getHeight()) / 2);
        ///int ovluvoff = ovlyoff / 4;

        char info[1000];  // general text buffer for various info messages

        Point2D<int> lastpointsent(w/2, h/2);


        // ######################################################################
        try {
                // let's do it!
                manager.start();

                // get the frame grabber to start streaming:
                gb->startStream();

                // initialize the timers:
                tim.reset(); masterclock.reset();
                p_tim.reset();

                Point2D<int> loc(84, 33);

                while(goforever)
                {
                        // grab image:
                        //Image< PixRGB<byte> > ima = gb->readRGB();
                        char filename[255];

                        sprintf(filename, "/usr/home/elazary/images/backyard/2/dframe%06d.ppm", frame+1);
                        Image< PixRGB<byte> > ima = rescale(Raster::ReadRGB(filename), w, h);
                        usleep(10000);

                        // display image:
                        inplacePaste(disp, ima, Point2D<int>(0, 0));
                        Image<float> dispsm = rescale(sm, w,h) * SMFAC;
                        inplaceNormalize(dispsm, 0.0F, 255.0F);

                        //inplacePaste(disp, Image<PixRGB<byte> >
                        //            (toRGB(quickInterpolate(dispsm, 1 << sml))),
                        //           Point2D<int>(w, 0));


                        inplacePaste(disp, Image<PixRGB<byte> >(toRGB(dispsm)), Point2D<int>(w, 0));


                        /*sc->evolve(masterclock.getSimTime(), (Retina*) 0);
                                Point2D<int> eye = sc->getDecision(masterclock.getSimTime());
                                if (eye.i >= 0) fixation = eye; */

                        Point2D<int> fix2(fixation); fix2.i += w;
                        if (fixation.i >= 0)
                        {
                                drawDisk(disp, fixation, foa_size/6+2, PixRGB<byte>(20, 50, 255));
                                drawDisk(disp, fixation, foa_size/6, PixRGB<byte>(255, 255, 20));
                                drawDisk(disp, fix2, foa_size/6+2, PixRGB<byte>(20, 50, 255));
                                drawDisk(disp, fix2, foa_size/6, PixRGB<byte>(255, 255, 20));
                        }



                        xwin.drawImage(disp);

                        //get the location of the mouse click, find the highst local value, and set that as the bias
                        // point
                        //Point2D<int> loc = xwin.getLastMouseClick();
                        if (loc.isValid()){
                                LINFO("Loc: %i %i\n", loc.i, loc.j);

                                //descard the currently processed saliency map
                                //build a new unbised saliency map, and find the highest local value within it
                                while(!smt->outputReady()){                //wait for any processing to finish
                                        usleep(100);
                                }

                                smt->setBiasSM(false);        //let the system know we dont want a biased smap
                                smt->setSaliencyMapLevel(0); //set the saliency map level to 0
                                smt->newInput(ima); //set the image

                                while(!smt->outputReady()){ //wait for salieny map
                                        usleep(100);
                                }

                                Image<float> tmpSM = smt->getOutput();


                                Image<float> smp;

                                for(int i=0; i<14; i++){
                                        if (smt->cmaps[i].initialized()){
                                                //inplaceNormalize(smt->cmaps[i], 0.0F, 255.0F);
                                                //xwin2.drawImage(smt->cmaps[i]);
                                                Point2D<int> center_loc(loc.i-((WINSIZE-1)/2), loc.j-((WINSIZE-1)/2));

                                                Image<float> target = crop(smt->cmaps[i],center_loc, Dims(WINSIZE,WINSIZE));
                                                bias[i] = target;
                                                xwin3.drawImage(target);

                                                LINFO("Convol");
                                                for(int y=0; y<target.getHeight()/2; y++){
                                                        for(int x=0; x<target.getWidth()/2; x++){
                                                                float tmp = target.getVal(x,y);
                                                                target.setVal(x,y,target.getVal(target.getWidth()-1-x,target.getHeight()-1-y));
                                                                target.setVal(target.getWidth()-1-x,target.getHeight()-1-y,tmp);

                                                        }
                                                }

                                                /*
                                                         for(int x=0; x<target.getHeight()/2; x++){
                                                         for(int y=0; y<target.getWidth()/2; y++){
                                                         float tmp = target.getVal(x,y);
                                                         target.setVal(x,y,target.getVal(x,target.getHeight()-1-y));
                                                         target.setVal(x,target.getHeight()-1-y,tmp);
                                                         }
                                                         }*/
                                                xwin2.drawImage(target);
                                                getchar();

                                                Convolver fc(target, smt->cmaps[i].getDims());
                                                Image<float> conv1 = fc.fftConvolve(smt->cmaps[i]);
                                                float maxval; Point2D<int> currwin; findMin(conv1, currwin, maxval);
                                                LINFO("Min at %ix%i", currwin.i, currwin.j);
                                                xwin3.drawImage(conv1);
                                                getchar();

                                        }
                                }

                                smt->setSMBias(bias);

                                smt->setBiasSM(true);        //let the system know we want a biased smap

                                smt->newInput(ima); //set the image to be processed
                                loc.i = -1; loc.j = -1;

                        }


                        // are we ready to process a new frame? if so, send our new one:
                        if (smt->outputReady())
                        {
                                // let's get the previous results, if any:
                                Image<float> out = smt->getOutput();
                                if (out.initialized())
                                        sm = out;

                                // find most salient location and feed saccade controller:
                                float maxval; Point2D<int> currwin; findMin(sm, currwin, maxval);
                                //since the saliency map is smaller by the winsize, remap the winner point
                                currwin.i = currwin.i + (WINSIZE-1)/2;
                                currwin.j = currwin.j + (WINSIZE-1)/2;

                                if (currwin.isValid()) fixation = currwin;

                                LINFO("Winner:%ix%i", fixation.i, fixation.j);
                                //        getchar();

                                double dist = 0;
                                for(int i=0; i<14; i++){
                                        if (smt->cmaps[i].initialized()){
                                                if (bias[i].initialized() && newBias[i].initialized()){
                                                        Point2D<int> center_fixation(fixation.i-((WINSIZE-1)/2),
                                                                        fixation.j-((WINSIZE-1)/2));
                                                        Image<float> target = crop(smt->cmaps[i],
                                                                        center_fixation,
                                                                        Dims(WINSIZE,WINSIZE));
                                                        newBias[i] = target;
                                                        dist += distance(bias[i], newBias[i]);
                                                        bias[i] = newBias[i];
                                                }
                                        }
                                }
                                LINFO("Distance: %f", dist);
                                if (dist < 2000)
                                        smt->setSMBias(newBias);
                                //drawDisk(ima, newwin.p, 10, PixRGB<byte>(255, 0, 0));

                                // feed our current image as next one to process:
                                //smt->newInput(decXY(ima));
                                smt->newInput(ima);
                                lastframesent = frame;
                                //LINFO("Processing frame %u", frame);

                                // compute and show framerate and stats for the frame we have processed
                                //over the last NAVG frames:
                                p_avgtime += p_tim.getReset(); p_avgn ++;
                                if (p_avgn == NAVG)
                                {
                                        p_fps = 1000.0F / float(p_avgtime) * float(p_avgn);
                                        p_avgtime = 0; p_avgn = 0;
                                }

                        }

                        // compute and show framerate and stats over the last NAVG frames:
                        avgtime += tim.getReset(); avgn ++;
                        if (avgn == NAVG)
                        {
                                fps = 1000.0F / float(avgtime) * float(avgn);
                                avgtime = 0; avgn = 0;
                        }

                        // create an info string:
                        sprintf(info, "%06u/%06u %.1ffps/%.5ffps   ",
                                        frame, lastframesent, fps, p_fps);

                        writeText(disp, Point2D<int>(0, h), info,
                                        PixRGB<byte>(255), PixRGB<byte>(127));

                        // ready for next frame:
                        ++ frame;
                }

                // get ready to terminate:
                manager.stop();
                close(sock);

        } catch ( ... ) { };

        return 0;
}

#endif
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
