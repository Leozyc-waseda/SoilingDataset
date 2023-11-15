/*!@file Nerdcam/nerd-cam-wan.C [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Nerdcam/nerd-cam-wan.C $
// $Id: nerd-cam-wan.C 6454 2006-04-11 00:47:40Z rjpeters $

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>

const char* saveFile;

//! Does a single image grab and writes to file specified
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("NerdCam WAN");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  // choose a V4Lgrabber by default, and a few custom grabbing
  // defaults, for backward compatibility with an older version of
  // this program:
  manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
  manager.setOptionValString(&OPT_FrameGrabberDims, "320x240");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<file>", 1, 1) == false) return(1);

  // do post-command-line configs:
  saveFile = manager.getExtraArg(0).c_str();

  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");

  // let's get all our ModelComponent instances started:
  manager.start();

  Image<PixRGB<byte> > nerd;
  //LINFO("GRABBING %s",saveFile);
  for(int i = 0 ; i < 16 ; i++)
    nerd = gb->readRGB();
  //LINFO("RESTERING %s", saveFile);
  Raster::WriteRGB(nerd,sformat("%snerd-cam-WAN.ppm",saveFile));
  Raster::WriteGray(luminance(nerd),
                    sformat("%snerd-cam-WAN.grey.pgm",saveFile));

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
