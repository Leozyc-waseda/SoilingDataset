/*
 *  test-BOWN.C
 *
 *
 *  Created by Randolph Voorhies on 11/25/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

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
#include "Image/ShapeOps.H"
#include "V2.H"
#include "V1.H"
#include <vector>
#include <cstdio>

using namespace std;

int main(int argc, char* argv[]) {

        ModelManager manager("Border Ownership");

        nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
        manager.addSubComponent(ifs);
        manager.exportOptions(MC_RECURSE);

        //Parse command-line:
        if (manager.parseCommandLine(argc, argv,"[image {*.ppm}]", 0, 1) == false) return(1);

        //Start the model manager
        manager.start();

        Image < float > jImage(Dims(50,50), ZEROS);        //The input image

        rutz::shared_ptr<XWinManaged> window1;
        window1.reset(new XWinManaged(Dims(ifs->getWidth()*2, ifs->getHeight()*2), 0, 0, "Border Ownership"));

        V1 myV1(Dims(100,100));
        V2 myV2(Dims(100,100));

        Image<byte> inputImage;
        Image<byte> edgeImage;
        Image<float> orientationImage;
        Image<float> magnitudeImage(ifs->peekDims(), ZEROS);      //First Derivative Magnitude Image

        inputImage.resize(ifs->peekDims());
        edgeImage.resize(ifs->peekDims());
        orientationImage.resize(ifs->peekDims());

        LINFO("Input Image = (%d,%d)", ifs->peekDims().w(), ifs->peekDims().h());

        while(1) {
                inputImage = ifs->readGray();
                edgeImage.clear();
                orientationImage.clear();
                //myV1.cannyEdgeDetect(inputImage,edgeImage,orientationImage);

                Image<float> fblurImage = luminance(convGauss(inputImage, sigma, sigma,10));

                //Find the magnitude and orientation
                gradient(fblurImage, magnitudeImage, orientationImage);

                window1->drawImage(inputImage,0,0);
                window1->drawImage(orientationImage,edgeImage.getWidth(),0);
        /*        window1->drawImage(scaledJImage,0,0);
                window1->drawImage(scaledWImage,scaledJImage.getWidth(), 0);*/
                Raster::waitForKey();

        }

}
