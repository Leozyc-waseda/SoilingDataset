/*!@file Corba/tests/EchoImageClient.C a test program to send an image using corba:
  This is the client that sends and receives the image  */

//////////////////////////////////////////////////////////////////// //
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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/tests/EchoImageClient.C $
// $Id: EchoImageClient.C 6795 2006-06-29 20:45:32Z rjpeters $
//




#include <iostream>
#include "GUI/XWinManaged.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Image/ColorOps.H"
#include "SIFT/ScaleSpace.H"
#include "Raster/Raster.H"
#include <time.h>
#include "Corba/Objects/EchoImageServerSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "Corba/CorbaUtil.H"


int main(int argc, char **argv)
{
  CORBA::ORB_ptr orb = CORBA::ORB_init(argc,argv,"omniORB4");

  CORBA::Object_ptr objEchoImageRef[10]; int nEchoImageObj;
  if (!getMultiObjectRef(orb, "saliency.EchoImage",
                         objEchoImageRef, nEchoImageObj))
    {
      LFATAL("Can not find any object to bind with");
    }

  EchoImageServer_var echoImage =
    EchoImageServer::_narrow(objEchoImageRef[0]);

  Image< PixRGB<byte> > image;
  if (argc < 2)
    LFATAL("USAGE: EchoImageClient <image>");
  else
    image = Raster::ReadRGB(argv[1]);

  LINFO("Sending image size %ix%i", image.getWidth(), image.getHeight());
  //envoke the methods from the object
  ImageOrb *orbImg = echoImage->echo(*image2Orb(image));
  LINFO("Got ImageOrb");

  //convert the result to an Image
  Image< PixRGB<byte> > image2;
  orb2Image(*orbImg, image2);
  LINFO("Convert ImgOrb to image size %ix%i",
        image2.getWidth(), image2.getHeight());

  orb->destroy();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
