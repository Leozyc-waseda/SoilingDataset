/*test-Attentionater.C*/

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
#include "Neuro/EnvVisualCortex.H"
#include "Image/ShapeOps.H"
#include "Image/MathOps.H"
#include "Util/MathFunctions.H"
#include "SeaBee/Attentionator.H"


int main(int argc, char* argv[])
{

  MYLOGVERB = LOG_INFO;

  ModelManager manager("Attentionater Tester");

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<Attentionator> attentionator(new Attentionator(manager));
  manager.addSubComponent(attentionator);

 // nub::soft_ref<EnvVisualCortex> EVC(new EnvVisualCortex(manager));
  //manager.addSubComponent(EVC);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "[image {*.ppm}]",
                               0, 1)
      == false) return(1);


  manager.setOptionValString(&OPT_InputFrameDims, convertToString(ifs->peekDims()));

  manager.setModelParamVal("InputFrameDims", ifs->peekDims(),
                           MC_RECURSE | MC_IGNORE_MISSING);

  manager.start();
  int w = ifs->getWidth(),  h = ifs->getHeight();
  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);


  bool goforever = true;

  rutz::shared_ptr<XWinManaged> dispWin;
  dispWin.reset(new XWinManaged(Dims(w*2,h*2), 0, 0, "test-Attentionator Display"));

  // input and output image
  Image< PixRGB<byte> > img(w,h, NO_INIT);

        while(goforever)
        {
                Image< PixRGB<byte> > dispImg(w*2,h*2, ZEROS);

                ifs->updateNext();
                img = ifs->readRGB();
                attentionator->updateImage(img);

                if(!img.initialized()) {Raster::waitForKey(); break; }

                inplacePaste(dispImg, img, Point2D<int>(0,0));
                inplacePaste(dispImg, attentionator->getSaliencyMap(), Point2D<int>(0,img.getHeight()));
                drawCross(img, attentionator->getSalientPoint(), PixRGB <byte> (0, 255, 0), 5, 3);
                inplacePaste(dispImg, img, Point2D<int>(w,0));

                dispWin->drawImage(dispImg, 0, 0);
        }
        Raster::waitForKey();
        // get ready to terminate:
        manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
