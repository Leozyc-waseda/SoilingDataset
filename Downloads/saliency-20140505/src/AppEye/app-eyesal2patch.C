/*!@file AppMedia/stim-eyesal2patch.C image patches from aneyesal file */
//takes as input an eyesal file, a movie path and a frame duration

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppEye/app-eyesal2patch.C $
// $Id: app-eyesal2patch.C 10794 2009-02-08 06:21:09Z itti $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Util/MathFunctions.H"
#include "Image/Pixels.H"
#include "Image/DrawOps.H"
#include "Raster/Raster.H"
#include "Raster/PngWriter.H"
#include "Transport/FrameIstream.H"
#include "Image/Rectangle.H"
#include "Media/MpegInputStream.H"
#include "Media/MediaOpts.H"
#include "GUI/XWindow.H"
#include "Raster/GenericFrame.H"
#include "Psycho/EyesalData.H"
#include "Util/StringConversions.H"

#include <vector>
#include <fstream>

#define RAD 62

int main(const int argc, const char **argv)
{


  ModelManager *mgr = new ModelManager("eyesal2patch");

//hook up mpeg stream
  nub::ref<InputMPEGStream> ifs(new InputMPEGStream(*mgr));
  mgr->addSubComponent(ifs);
  mgr->exportOptions(MC_RECURSE);

//get some command line stuff
  std::string eyesalFile = argv[1];//filename
  std::string moviePath = argv[2];//moviepath
  const float fdur = fromStr<float>(argv[3]);

  EyesalData ed(eyesalFile);//load our eyesal file

  // Parse command-line:
  if (mgr->parseCommandLine(argc, argv, "", 3, 3) == false) return(1);

  // let's get all our ModelComponent instances started:
  mgr->start();

  //set up a preview/demo window
  Dims screen(800,600);
  XWindow win(screen, 0, 0, "Eyesal2patch");//preview window

  std::string mname = "";
  std::string fname = "";


  //ok lets, loop through all our saccades
  for (size_t ii = 0; ii < ed.size(); ii++){

      //setup our output file
      std::string mtfname =  ed.getFileName(ii);
      size_t idx = mtfname.rfind('.');
      std::string outname =  mtfname.substr(0,idx) + "-"  + convertToString(ii) + ".png";

      if (ed.getFileName(ii).compare(mname) != 0) {
          LINFO("Movie changed::%s",mname.c_str());
          //set our mpeg source
          size_t idp = mtfname.find('-');
          mname = mtfname.substr(idp+1);
          idx = mname.rfind('.');
          ifs->setFileName(moviePath+mname.substr(0,idx)+".mpg");
      }

      Image< PixRGB<byte> > input;


    //go to the frame of the saccade
      int fnum = (int)ceil((ed.getTime(ii) / fdur) - 1.0F);
      if (!ifs->setFrameNumber(fnum))
        break;

          //grab the image
      input = ifs->readRGB();
      if (!input.initialized())
          break;

      Image< PixRGB<byte> > bg(800,600,ZEROS);
      inplacePaste(bg,input,Point2D<int>(79,59));


      LINFO("\nSaccade on Frame %d",fnum);


          //display the image with saccade location on screen
      Point2D<int> cp = ed.getXYpos(ii);
      cp.i+=79;
      cp.j+=59;

      Rectangle rwin = Rectangle::tlbrO(cp.j-RAD,cp.i-RAD,cp.j+RAD,cp.i+RAD);


      if (bg.rectangleOk(rwin)){    //only if the point is valid

              //grab our image for output
          Image< PixRGB<byte> > output = crop(bg,rwin);
          PngWriter::writeRGB(output,outname);//output

          drawRect(bg,rwin,PixRGB<byte>(0,255,0),2);//paint it

      }
      else {
          LINFO ("Rectangle out of bounds");
          drawDisk(bg,cp,5,PixRGB<byte>(0,255,0));    //if we arent in range
      }

          //some notes on screen
      char str[25];
      sprintf(str,"Saccade: %d\nTime: %g",(int)ii,ed.getTime(ii));
      writeText(bg,Point2D<int>(2,0),str,
                PixRGB<byte>(255,0,0),PixRGB<byte>(0,0,0), SimpleFont::FIXED(9));
      win.drawImage(bg,0,0);

          //sleep(1);

  }
      //stop all our modelcomponents
  mgr->stop();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */






