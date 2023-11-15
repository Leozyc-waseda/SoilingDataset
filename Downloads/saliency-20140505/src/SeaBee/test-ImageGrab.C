/*test-ImageGrab.C   */

#include "Component/ModelManager.H"

#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Image/CutPaste.H"

#include "BeoSub/IsolateColor.H"
#include "Image/DrawOps.H"
#include "Image/ColorOps.H"

#include "GUI/XWinManaged.H"
//#include "Image/OpenCVUtil.H"

//#include "SeaBee/PipeRecognizer.H"


int main(int argc, char* argv[])
{

  MYLOGVERB = LOG_INFO;

  ModelManager manager("PipeRecognizer Tester");

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "[image {*.ppm}]",
                               0, 1)
      == false) return(1);

  int w = ifs->getWidth(),  h = ifs->getHeight();
  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);
  manager.setOptionValString(&OPT_InputFrameDims, dims);

  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  manager.start();

  bool goforever = true;

  rutz::shared_ptr<XWinManaged> dispWin;
  dispWin.reset(new XWinManaged(Dims(w*2,h*2), 0, 0, "Pipe Recognizer Display"));

  // input and output image

  Image< PixRGB<byte> > img(w,h, ZEROS);
  Image< PixRGB<byte> > canny(w,h, ZEROS);

  //uint fNum = 0;
  while(goforever)
    {
      Image< PixRGB<byte> > dispImg(w*2,h*2, ZEROS);
      rutz::shared_ptr<Image< PixRGB<byte> > > outputImg(new Image<PixRGB<byte> >(w,h, ZEROS));

      //rutz::shared_ptr<Image<byte> > orangeIsoImage;
      //orangeIsoImage.reset(new Image<byte>(w,h, ZEROS));

      ifs->updateNext(); img = ifs->readRGB();
      if(!img.initialized()) {Raster::waitForKey(); break; }

          //find edges of segmented image using canny
          //IplImage *edge = cvCreateImage( cvGetSize(img2ipl(img)), 8, 1 );
          //cvCanny( img2ipl(luminance(img)), edge, 100, 150, 3 );//150,200,3
         // canny = ipl2rgb(edge);

      //inplacePaste(dispImg, canny, Point2D<int>(w,0));
      inplacePaste(dispImg, img, Point2D<int>(0,0));

      //orangeIsoImage->resize(w,h);
      //isolateOrange(img, *orangeIsoImage);

      //inplacePaste(dispImg, toRGB(*orangeIsoImage), Point2D<int>(w,0));

      /*pipeRecognizer->getPipeLocation(orangeIsoImage,
                                      outputImg,
                                      PipeRecognizer::HOUGH,
                                      pipeCenter,
                                      pipeAngle);

      projPoint.i = (int)(pipeCenter->i+30*cos(*pipeAngle));
      projPoint.j = (int)(pipeCenter->j+30*sin(*pipeAngle));

      drawLine(*outputImg, *pipeCenter, projPoint, PixRGB <byte> (255, 255,0), 3);

      inplacePaste(dispImg, *outputImg, Point2D<int>(0,h));
 */

      //wait a little

      dispWin->drawImage(dispImg, 0, 0);
      Raster::waitForKey();
    }

  // get ready to terminate:
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
