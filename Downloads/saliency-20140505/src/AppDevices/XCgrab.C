/*!@file AppPsycho/videograb.C grab frames and save the debayered color
  images to disk. Use it like: XCgrab <name>*/

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
// Primary maintainer for this file: Zhicheng Li <zhicheng@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/XCgrab.C $
// $Id: XCgrab.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/DeviceOpts.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/ShapeOps.H"
#include "Image/CutPaste.H"
#include "Image/ImageCache.H"
#include "Image/Pixels.H"
#include "GUI/SDLdisplay.H"
#include "GUI/GUIOpts.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "Video/RgbConversion.H" // for toVideoYUV422()
#include "Raster/DeBayer.H" // for debayer()

#include <pthread.h>

//! number of frames over which frame rate is computed
#define NAVG 20

#define MAXSAVETHREAD 4

//! Enumerate the video preview type
enum PreviewType
  {
    PrevAll,
    PrevTopLeft,
    PrevTopRight,
    PrevBotLeft,
    PrevBotRight,
    PrevCenter
  };

//! counter for saved frame number
uint fnb = 0;
pthread_mutex_t qmutex_cache, qmutex_cacheRGB, qmutex_imaShow;
pthread_mutex_t qmutex_mgz;
ImageCache<byte> cache;
ImageCache<PixRGB<byte> > cacheRGB;
std::vector<std::string> base;
bool saving = false;
Image<PixRGB<byte> > imaShow;
PreviewType prevType = PrevCenter;
Dims showDim;

//save the debayered frames in disk as "ppm" format
static void* saveframes(void *)
{
  while(1) {
    Image<PixRGB<byte> > imaRGB; bool havemore = false;
    uint cntTmp = 0;

    // do we have images ready to go?
    pthread_mutex_lock(&qmutex_cacheRGB);
    if (cacheRGB.size()){
      imaRGB = cacheRGB.pop_front();
      fnb++;
      cntTmp = fnb;
    }
    if (cacheRGB.size()) havemore = true;
    pthread_mutex_unlock(&qmutex_cacheRGB);

    // if we got an image, save it:
    if (imaRGB.initialized())
      {
        // we save each frame to a different base in a rolling manner:
        const char *b = base[cntTmp % base.size()].c_str();
        Raster::WriteRGB(imaRGB, sformat("%s%06u.ppm", b, fnb));
      }

    if (havemore == false) usleep(200);
  }
  return NULL;
}

//debayer the captured image and display a smaller size on the screen
static void* debayerframes(void *)
{
  while(1){
    Image<byte> ima; bool havemore = false;
    Image<PixRGB<byte> > imaRGB;

    pthread_mutex_lock(&qmutex_cache);
    if(cache.size()) ima = cache.pop_front();
    if(cache.size()) havemore = true;
    pthread_mutex_unlock(&qmutex_cache);

    if( ima.initialized())
      {
        imaRGB = deBayer(ima, BAYER_GBRG);
        pthread_mutex_lock(&qmutex_imaShow);

        switch(prevType){
        case PrevAll:  // scaled size of the whole image
          imaShow = rescale(imaRGB, showDim.w(), showDim.h(),
                            RESCALE_SIMPLE_NOINTERP);
          break;
        case PrevTopLeft:  // top left
          imaShow = crop(imaRGB, Point2D<int>(0,0), showDim);
          break;
        case PrevTopRight:  // top right
          imaShow = crop(imaRGB, Point2D<int>
                         (imaRGB.getWidth()-showDim.w(),0), showDim);
          break;
        case PrevBotLeft:  // bottom left
          imaShow = crop(imaRGB, Point2D<int>
                         (0,imaRGB.getHeight()-showDim.h()), showDim);
          break;
        case PrevBotRight:  // bottom right
          imaShow = crop(imaRGB, Point2D<int>
                         (imaRGB.getWidth()-showDim.w(),
                          imaRGB.getHeight()-showDim.h()), showDim);
          break;
        case PrevCenter:  // center
          imaShow = crop(imaRGB, Point2D<int>
                         ((imaRGB.getWidth()-showDim.w())/2,
                          (imaRGB.getHeight()-showDim.h())/2),showDim);
          break;
        default:
          LFATAL("the preview type should between 0 and 5, now is %d", prevType);
        }

        pthread_mutex_unlock(&qmutex_imaShow);

        if(saving)
          {
            pthread_mutex_lock(&qmutex_cacheRGB);
            cacheRGB.push_back(imaRGB);
            pthread_mutex_unlock(&qmutex_cacheRGB);
          }
      }

    if(havemore == false) usleep(20);
  }
  return NULL;
}


/*! This simple executable grabs video frames through the EPIX XC HD
  camera link grabber (see XCgrabber.H). Selection of the grabber type
  is made via the  --fg-type=XC command-line option. Frames are pushed
  into a queue and  a second thread then tries to empty the queue as
  quickly as possible by writing the frames to disk. In testing, PPM
  format actually gave better frame rates than PNG, so that's what is
  used.  Press <SPACE> to start grabbing and <SPACE> again to stop. */
static int submain(const int argc, char** argv)
{
  // instantiate a model manager:
  ModelManager manager("Frame Grabber");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<SDLdisplay> d(new SDLdisplay(manager));
  manager.addSubComponent(d);

  manager.setOptionValString(&OPT_SDLdisplayPriority, "0");
  manager.setOptionValString(&OPT_FrameGrabberType, "XC");
  manager.setOptionValString(&OPT_SDLdisplayFullscreen,"false");
  manager.setOptionValString(&OPT_SDLdisplayDims, "960x640");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<basename> ... <basename>",
                               1, MAXSAVETHREAD) == false)
    return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to have XC camera and XClibrary");

  // get the basename:
  for (uint i = 0; i < manager.numExtraArgs(); i ++)
    base.push_back(manager.getExtraArg(i));

  // let's get all our ModelComponent instances started:
  manager.start();

  // get ready for main loop:
  Timer tim,timer; uint64 t[NAVG]; int frame = 0;
  d->clearScreen(PixRGB<byte>(128)); bool doit = true;

  showDim = d->getDims();
  int iw = showDim.w(), ih = showDim.h();
  int dw = d->getDims().w(), dh = d->getDims().h();

  int ovlyoff = (dw - iw) / 2 + dw * ((dh - ih) / 2);
  int ovluvoff = (dw - iw) / 4 + dw * ((dh - ih) / 8);
  int ystride = (dw - iw), uvstride = (dw - iw) / 2;

  pthread_t saver[MAXSAVETHREAD];
  for(int ii = 0; ii<(int)base.size(); ii++)
    pthread_create(saver+ii, NULL, &saveframes, (void *)NULL);

  pthread_t debayer1;
  pthread_create(&debayer1, NULL, &debayerframes, (void*) NULL);

  // create an overlay:
  d->createYUVoverlay(SDL_YV12_OVERLAY);

  // get the frame grabber to start streaming:
  gb->startStream();

  // main loop:
  // if we type space to record then next space will stop the first record section
  // the next section will be begin when the third space typed and so on..
  //int section_cnt = 0;
  float frate = 0.0f;
  while(doit) {
    tim.reset();

    // grab a frame:
    Image<byte> ima = gb->readGray();
    pthread_mutex_lock(&qmutex_cache);
    cache.push_back(ima);
    pthread_mutex_unlock(&qmutex_cache);

    // to measure display time:
    uint64 t0 = tim.get();

    // if saving, push image into queue:
    pthread_mutex_lock(&qmutex_imaShow);
    if (saving)
      {
        const std::string msg =
          sformat(" %.1ffps [%04d] ", frate, cacheRGB.size());
        writeText(imaShow, Point2D<int>(0, 0), msg.c_str());
      }
    else // tell user we are ready to save
      {
        const std::string msg =
          sformat(" [SPC] to save %.1ffp [%04d] ", frate, cacheRGB.size());
        writeText(imaShow, Point2D<int>(0, 0), msg.c_str());
      }

    // show the frame:
    SDL_Overlay* ovl = d->lockYUVoverlay();
    toVideoYUV422(imaShow, ovl->pixels[0] + ovlyoff,
                  ovl->pixels[2] + ovluvoff,
                  ovl->pixels[1] + ovluvoff,
             ystride, uvstride, uvstride);

    d->unlockYUVoverlay();
    d->displayYUVoverlay(-1, SDLdisplay::NO_WAIT);
    pthread_mutex_unlock(&qmutex_imaShow);

    // check for space bar pressed; note: will abort us violently if
    // ESC is pressed instead:
    int ii = (d->checkForKey());
    if(ii == ' ')
      { saving = ! saving;
        prevType = PrevAll;
      }
    else if(ii == 'q')
      prevType = PrevTopLeft;
    else if(ii == 'e')
      prevType = PrevTopRight;
    else if(ii == 'a')
      prevType = PrevBotLeft;
    else if(ii == 'd')
      prevType = PrevBotRight;
    else if(ii == 's')
      prevType = PrevCenter;
    else if(ii == 'w')
      prevType = PrevAll;

    t[frame % NAVG] = tim.get();
    t0 = t[frame % NAVG] - t0;
    if (t0 > 20000ULL) LINFO("Display took %lluus", t0);

    // compute and show framerate over the last NAVG frames:
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
        frate = 1000.0F / float(avg) * float(NAVG);
        LINFO("Frame rate %f fps, buf size %u, time %f", frate,
              cacheRGB.size(), timer.getSecs());
      }
    frame ++;
  }
  d->destroyYUVoverlay();
  manager.stop();

  // all done!
  return 0;
}

extern "C" int main(const int argc, char** argv)
{
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
