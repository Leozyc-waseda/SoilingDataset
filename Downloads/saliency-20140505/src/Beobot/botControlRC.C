/*!@file Beobot/BotControlRC.C a test program run the botcontroller */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/botControlRC.C $
// $Id: botControlRC.C 10794 2009-02-08 06:21:09Z itti $
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
#include "Corba/Objects/BotControlSK.hh"
#include "Neuro/EnvVisualCortex.H"


#define UP  98
#define DOWN 104
#define LEFT 100
#define RIGHT 102


//////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{


  try {
    MYLOGVERB = LOG_INFO;
    ModelManager *mgr = new ModelManager("Test ObjRec");

    CORBA::ORB_var orb = CORBA::ORB_init(argc, argv);


    CORBA::Object_ptr objBotControlRef[10]; int nBotControlObj;
    if (!getMultiObjectRef(orb, "botControl.irobot",
          objBotControlRef, nBotControlObj))
    {
      printf("Can not find any object to bind with\n");
      return 1;
    } else {
      printf("Found %i object, binding to the last one\n", nBotControlObj);
    }

    BotControl_var botControl = BotControl::_narrow(objBotControlRef[nBotControlObj-1]);

    if( CORBA::is_nil(botControl) ) {
      printf("Can't narrow reference to type Echo (or it was nil).\n");
      return 1;
    }


    nub::ref<EnvVisualCortex> evc(new EnvVisualCortex(*mgr));
    mgr->addSubComponent(evc);
    mgr->exportOptions(MC_RECURSE);

    if (mgr->parseCommandLine(
          (const int)argc, (const char**)argv, "", 0, 0) == false)
      return 1;

    mgr->start();


    XWinManaged* xwin = new XWinManaged(Dims(320*2,240), -1, -1, "Bot Control");
    Image<PixRGB<byte> > img;

    float currentSpeed = 0;
    float currentSteering = 0;
    while(true)
    {
      ImageOrb *orbImg = botControl->getImageSensor(0);
      if (orbImg != NULL)
      {
        orb2Image(*orbImg, img);
      }
      img = rescale(img, 320, 240);

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
          currentSpeed += 10;
          botControl->setSpeed(currentSpeed);
          currentSteering = 0;
          botControl->setSteering(0);
          break;
        case DOWN:
          currentSpeed -= 10;
          botControl->setSpeed(currentSpeed);
          currentSteering = 0;
          botControl->setSteering(0);
          break;
        case LEFT:
          currentSteering +=10;
          botControl->setSpeed(100);
          botControl->setSteering(currentSteering);
          break;
        case RIGHT:
          currentSteering -=10;
          botControl->setSpeed(100);
          botControl->setSteering(currentSteering);
          break;
        case 65:  // space
          currentSpeed = 0;
          currentSteering = 0;
          botControl->setSteering(currentSteering);
          botControl->setSpeed(currentSpeed);
          break;
        case 33: // p
          botControl->playSong(0); //play the imerial march song
          break;
        case 40: //d
          botControl->dock(); //dock with charging station
          break;
        case 27: //r
          botControl->setOIMode(131); //safe mode
          break;

        default:
            if (key != -1)
               printf("Unknown Key %i\n", key);
            break;

      }

      usleep(10000);

    }

    orb->destroy();
  }
  catch(CORBA::TRANSIENT&) {
    printf("Caught system exception TRANSIENT -- unable to contact the server \n");
  }
  catch(CORBA::SystemException& ex) {
    printf("Caught a CORBA::%s\n", ex._name());
  }
  catch(CORBA::Exception& ex) {
    printf("Caught CORBA::Exception: %s\n", ex._name());
  }
  catch(omniORB::fatalException& fe) {
    printf("Caught omniORB::fatalException:\n");
    printf("  file: %s", fe.file());
    printf("  line: %i", fe.line());
    printf("  mesg: %s", fe.errmsg());
  }
  return 0;
}
