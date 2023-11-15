/*!@file SalGlasses/spyGlasses.C a test program run the SalGlasses */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SalGlasses/spyGlasses.C $
// $Id: spyGlasses.C 10794 2009-02-08 06:21:09Z itti $
//

#include <stdio.h>
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/ShapeOps.H"
#include "Image/DrawOps.H"
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include "Image/Layout.H"
#include "GUI/XWinManaged.H"
#include "Corba/ImageOrbUtil.H"
#include "Corba/CorbaUtil.H"
#include "Corba/Objects/SalGlassesSK.hh"
#include "Neuro/EnvVisualCortex.H"

#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <netdb.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <ctype.h>


#define UP  98
#define DOWN 104
#define LEFT 100
#define RIGHT 102

#define UDPPORT 9938


//////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{


  MYLOGVERB = LOG_INFO;
  ModelManager *mgr = new ModelManager("Test ObjRec");


  nub::ref<EnvVisualCortex> evc(new EnvVisualCortex(*mgr));
  mgr->addSubComponent(evc);
  mgr->exportOptions(MC_RECURSE);

  if (mgr->parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;

  mgr->start();

  //setup network comm
  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock == -1)
  {
    printf("Cannot create server socket\n");
    exit(0);
  }

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY); // accept from any host
  addr.sin_port = htons(UDPPORT);
  if (bind(sock, (struct sockaddr*)(&addr), sizeof(addr)) == -1)
  {
    printf("Cannot bind server socket\n");
    close(sock);
    exit(0);
  }

  printf("Lisening on port %i\n", UDPPORT);


  XWinManaged* xwin = new XWinManaged(Dims(320*2,240), -1, -1, "Saliency Glasses");
  Image<PixRGB<byte> > img;

  int width = 320, height = 240;
  char rvcBuf[1024+4];

  Image<PixRGB<byte> > img(width, height, NO_INIT);

  while(true)
  {

    struct sockaddr_in fromaddr; unsigned int fromlen = sizeof(fromaddr);
    int ret = recvfrom(sock, rvcBuf, , 0,
        (struct sockaddr *)(&fromaddr), &fromlen);
    printf("Received message %d bytes from %s:%d\n", ret,
        inet_ntoa(fromaddr.sin_addr), ntohs(fromaddr.sin_port));

    if (ret != 1024+4)
      printf("Invalid packet\n");
    int pos = *


    memcpy(img.getArrayPtr(), imgBuff, width*height*3);


    evc->input(img);

    Image<float> vcxMap = evc->getVCXmap();

    Point2D<int> maxPos; float maxVal;

    vcxMap = rescale(vcxMap, img.getDims());
    findMax(vcxMap, maxPos, maxVal);
    inplaceNormalize(vcxMap, 0.0F, 255.0F);

    drawCircle(img, maxPos, 20, PixRGB<byte>(255,0,0));

    Layout<PixRGB<byte> > outDisp;
    outDisp = vcat(outDisp, hcat(img, toRGB((Image<byte>)vcxMap)));
    xwin->drawRgbLayout(outDisp);


    int key = xwin->getLastKeyPress();

    switch (key)
    {
      case UP:
        break;
      case DOWN:
        break;
      case LEFT:
        break;
      case RIGHT:
        break;
      case 65:  // space
        break;
      case 33: // p
        break;
      case 40: //d
        break;
      case 27: //r
        break;
      default:
        if (key != -1)
          printf("Unknown Key %i\n", key);
        break;
    }
  }
return 0;
}
