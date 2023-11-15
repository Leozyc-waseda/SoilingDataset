/*
 *  test-RankOrderBorder.cpp
 *
 *
 *  Created by Randolph Voorhies on 11/11/07.
 *
 */

#define PI 3.14159265

//#include "Util/Types.H"
#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "GUI/XWinManaged.H"
#include "Image/ImageSet.H"
#include "Image/DrawOps.H"
#include "Image/PyrBuilder.H"
#include "Image/PyramidOps.H"
#include "Image/Kernels.H"
#include "pix.H"
#include "test-RankOrderBorder.H"
#include "roRetina.H"
#include "roUtil.H"
#include <queue>

using namespace std;





int main(int argc, char* argv[]) {

        ModelManager manager("Rank Order Border Ownership");
        nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
        manager.addSubComponent(ifs);
        manager.exportOptions(MC_RECURSE);

        //Parse command-line:
        if (manager.parseCommandLine(argc, argv,"[image {*.ppm}]", 0, 1) == false) return(1);

        //Start the model manager
        manager.start();

        rutz::shared_ptr < Image < float > > inputImage(new Image < float > (ifs->getWidth(), ifs->getHeight(), ZEROS));
        rutz::shared_ptr < Image < PixRGB<float> > > heatmap(new Image< PixRGB <float> >);
        heatmap->resize(ifs->getWidth(), ifs->getHeight());

        roRetina<float> retina;

        //Initialize the main display window
        int windowWidth = ifs->getWidth()*2;
        int windowHeight = ifs->getHeight()*2;
        rutz::shared_ptr<XWinManaged> window1;
        window1.reset(new XWinManaged(Dims(windowWidth,windowHeight), 0, 0, "Rank Order Border Ownership"));

        while(1) {
                //Update the input frame series, and read the image as an array of floats
                ifs->updateNext();
                *inputImage = ifs->readFloat();
                retina.update(inputImage);
                makeHeatmap<float>(retina.getSpikeWave(), heatmap);

                //Draw the output
                window1->drawImage(*inputImage,0,0);
                //drawCross<float>(centerSurroundImage, Point2D<int>(topPix.x, topPix.y), 100, 20);
                window1->drawImage(Image<PixRGB <byte> > (*heatmap),inputImage->getWidth(), 0);
        }
}
