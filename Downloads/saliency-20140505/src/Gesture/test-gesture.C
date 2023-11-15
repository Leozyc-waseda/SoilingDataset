/*!@file AppDevices/test-grab.C Test frame grabbing and X display */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-grab.C $
// $Id: test-grab.C 14290 2010-12-01 21:44:03Z itti $
//

#include "Util/log.H"

#if !defined(INVT_HAVE_OPENNI) || !defined(INVT_HAVE_NITE)

int main(const int argc, const char **argv)
{ LFATAL("Need OpenNI and Nite installed. Check configure output."); return 1; }

#else

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberFactory.H"
#include "Devices/OpenNIGrabber.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Raster/GenericFrame.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <XnOS.h>
#include <XnCppWrapper.h>

// Header for NITE
#include <XnVNite.h>

#define NAVG 20
// Callback for when the focus is in progress
void XN_CALLBACK_TYPE SessionProgress(const XnChar* strFocus, const XnPoint3D& ptFocusPoint, XnFloat fProgress, void* UserCxt)
{
	LINFO("Session in Progress!");
}
// callback for session start
void XN_CALLBACK_TYPE SessionStart(const XnPoint3D& ptFocusPoint, void* UserCxt)
{
	LINFO("Session started!");
}
// Callback for session end
void XN_CALLBACK_TYPE SessionEnd(void* UserCxt)
{
	printf("Session ended!");
}
// Callback for wave detection
void XN_CALLBACK_TYPE OnWaveCB(void* cxt)
{
	printf("Wave!\n");
}

//Callback for swipe left
void XN_CALLBACK_TYPE SwipeLeft(XnFloat fVelocity, XnFloat fAngle, void* cxt)
{
	printf("Swipe Left!\n");
}

//Callback for swipe right
void XN_CALLBACK_TYPE SwipeRight(XnFloat fVelocity, XnFloat fAngle, void* cxt)
{
	printf("Swipe Right!\n");
}

//Callback for swipe up
void XN_CALLBACK_TYPE SwipeUp(XnFloat fVelocity, XnFloat fAngle, void* cxt)
{
	printf("Swipe Up!\n");
}

//Callback for swipe down
void XN_CALLBACK_TYPE SwipeDown(XnFloat fVelocity, XnFloat fAngle, void* cxt)
{
	printf("Swipe Down!\n");
}

//Callback for steady
void XN_CALLBACK_TYPE Steady(XnFloat fVelocity, void* cxt)
{
	printf("Steady!\n");
}

//Callback for push
void XN_CALLBACK_TYPE Push(XnFloat fVelocity, XnFloat fAngle, void* cxt)
{
	printf("Push!\n");
}

int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("Gesture Tester");
  manager.exportOptions(MC_RECURSE);

  nub::soft_ref<OpenNIGrabber> gb = nub::soft_ref<OpenNIGrabber>(new OpenNIGrabber(manager));
  manager.addSubComponent(gb);
  gb->exportOptions(MC_RECURSE);
  // let's get all our ModelComponent instances started:
  
  manager.start();

  //NITE/Gesture
  xn::Context *gbNI;
  gbNI = gb->getContext();

	  XnVSessionManager *sessionManager = new XnVSessionManager();
	//the current session manager will look at the entire image for a start point.
	//there are some mechanisms that can help narrow the focus for   
	XnStatus rc = ((XnVSessionManager*)sessionManager)->Initialize(gbNI, "Click,Wave", "RaiseHand");
	if (rc != XN_STATUS_OK)
	{
		LFATAL("Session Manager couldn't initialize: %s\n", xnGetStatusString(rc));
		delete sessionManager;
	}
	gbNI->StartGeneratingAll();
 	 sessionManager->RegisterSession(NULL, &SessionStart, &SessionEnd, &SessionProgress);

	

	// init & register wave control
	XnVWaveDetector wc;
	wc.RegisterWave(NULL, OnWaveCB);
	sessionManager->AddListener(&wc);

	//init and register swipe control
	XnVSwipeDetector sw(false);
        sw.RegisterSwipeUp(NULL, SwipeUp);
        sw.RegisterSwipeDown(NULL, SwipeDown);
        sw.RegisterSwipeLeft(NULL, SwipeLeft);
        sw.RegisterSwipeRight(NULL, SwipeRight);
	sessionManager->AddListener(&sw);

	//init and register steady control
	XnVSteadyDetector st;
	st.RegisterSteady(NULL, Steady);
	sessionManager->AddListener(&st);

	//init and register push control
	XnVPushDetector pd;
	pd.RegisterPush(NULL, Push);
	sessionManager->AddListener(&pd);

  // get ready for main loop:
  Timer tim; uint64 t[NAVG]; int frame = 0;
  GenericFrameSpec fspec = gb->peekFrameSpec();
  Dims windims = fspec.dims;
  if (fspec.nativeType == GenericFrame::RGBD) windims = Dims(windims.w() * 2, windims.h());
  XWindow win(windims, -1, -1, "test-grab window");
  int count = 0;

  // prepare a gamma table for RGBD displays (e.g., Kinect grabber):
  uint16 itsGamma[2048];
  for (int i = 0; i < 2048; ++i) {
    float v = i/2048.0;
    v = powf(v, 3)* 6;
    itsGamma[i] = v*6*256;
  }

  // get the frame grabber to start streaming:
  gb->startStream();

  while(1) {
    ++count; tim.reset();

    GenericFrame fr = gb->readFrame();
    
    ((XnVSessionManager*)sessionManager)->Update(gbNI);

    Image< PixRGB<byte> > ima = fr.asRgbU8();

    if (fspec.nativeType == GenericFrame::RGBD) {
      Image<uint16> dimg = fr.asGrayU16(); // get the depth image

      Image<PixRGB<byte> > d(dimg.getDims(), NO_INIT);
      const int sz = dimg.size();
      for (int i = 0; i < sz; ++i) {
        uint v = dimg.getVal(i); if (v > 2047) v = 2047;
        int pval = itsGamma[v];
        int lb = pval & 0xff;
        switch (pval>>8) {
        case 0: d.setVal(i, PixRGB<byte>(255, 255-lb, 255-lb)); break;
        case 1: d.setVal(i, PixRGB<byte>(255, lb, 0)); break;
        case 2: d.setVal(i, PixRGB<byte>(255-lb, 255, 0)); break;
        case 3: d.setVal(i, PixRGB<byte>(0, 255, lb)); break;
        case 4: d.setVal(i, PixRGB<byte>(0, 255-lb, 255)); break;
        case 5: d.setVal(i, PixRGB<byte>(0, 0, 255-lb)); break;
        default: d.setVal(i, PixRGB<byte>(0, 0, 0)); break;
        }
      }
      ima = concatX(ima, d);
    }

    uint64 t0 = tim.get();  // to measure display time

    win.drawImage(ima);

    t[frame % NAVG] = tim.get();
    t0 = t[frame % NAVG] - t0;
    if (t0 > 20000ULL) LINFO("Display took %lluus", t0);


    // compute and show framerate over the last NAVG frames:
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
        float avg2 = 1000.0F / float(avg) * float(NAVG);
        printf("Framerate: %.1f fps\n", avg2);
      }
    frame ++;
  }

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
