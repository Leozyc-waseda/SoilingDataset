/*
 *  V1.C
 *
 *
 *  Created by Randolph Voorhies on 12/5/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef V1_H
#define V1_H


#include "Util/Types.H"
#include "Util/log.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Raster/Raster.H"
#include "rutz/shared_ptr.h"

#include <vector>

#define TRACE_CONTOUR_DEPTH_LIMIT 3
#define loThresh 10
#define hiThresh 16
#define sigma 2



class V1 {
public:
        V1(Dims d);
        void processInput(Image<byte> i);
        bool traceContour(const Image<float> &magnitudeImage, const int x, const int y, const int from, const int steps);
        void nonMaxSuppressAndContTrace(Image<byte> &edgeImage, const Image<float> magnitudeImage, const Image<float> orientationImage);
        void cannyEdgeDetect(const Image<byte> source, Image<byte> &edgeImage, Image<float> &orientationImage);
private:
        vector< Image<float> > layers;
        Image<byte> edgeImage;
        Dims itsDims;




};

V1::V1(Dims d) {
        itsDims = d;

        layers.resize(THETA_LEVELS, Image<float>(d, ZEROS));
}

void V1::processInput(Image<byte> i) {

}


//Depth limited search to trace contours
bool V1::traceContour(const Image<float> &magnitudeImage, const int x, const int y, const int from, const int steps) {
        if(magnitudeImage.getVal(x,y) >= hiThresh)
                return true;
        if(magnitudeImage.getVal(x,y) < loThresh)
                return false;
        if(x == 0 || x >= magnitudeImage.getWidth() - 1 || y == 0 || y >= magnitudeImage.getHeight() - 1)
                return false;

        if(steps >= TRACE_CONTOUR_DEPTH_LIMIT)
                return false;

        if(magnitudeImage.getVal(x-1,y) > loThresh && from != 1)
                if(traceContour(magnitudeImage, x-1, y, 5, steps+1))
                        return true;
        if(magnitudeImage.getVal(x-1,y-1) > loThresh && from != 2)
                if(traceContour(magnitudeImage, x-1, y-1, 6, steps+1))
                        return true;
        if(magnitudeImage.getVal(x-1,y+1) > loThresh && from != 8)
                if(traceContour(magnitudeImage, x-1, y+1, 4, steps+1))
                        return true;
        if(magnitudeImage.getVal(x,y-1) > loThresh && from != 3)
                if(traceContour(magnitudeImage, x, y-1, 7, steps+1))
                        return true;
        if(magnitudeImage.getVal(x,y+1) > loThresh && from != 7)
                if(traceContour(magnitudeImage, x, y+1, 3, steps+1))
                        return true;
        if(magnitudeImage.getVal(x+1,y-1) > loThresh && from != 4)
                if(traceContour(magnitudeImage, x+1, y-1, 8, steps+1))
                        return true;
        if(magnitudeImage.getVal(x+1,y) > loThresh && from != 5)
                if(traceContour(magnitudeImage, x+1, y, 1, steps+1))
                        return true;
        if(magnitudeImage.getVal(x+1,y+1) > loThresh && from != 6)
                if(traceContour(magnitudeImage, x+1, y+1, 2, steps+1))
                        return true;

        return false;
}

void V1::nonMaxSuppressAndContTrace(Image<byte> &edgeImage, const Image<float> magnitudeImage, const Image<float> orientationImage) {
        float mag, pa=0, pb=0, orientDeg;

        for(int x = 1; x < magnitudeImage.getWidth() - 1; x++)
                for(int y = 1; y < magnitudeImage.getHeight() - 1; y++) {
                        mag = magnitudeImage.getVal(x,y);
                        orientDeg = orientationImage.getVal(x,y)*(180/PI);

                        if((orientDeg >= 0 && orientDeg <= 45) || (orientDeg > -180 && orientDeg <= -135)) {
                                pa = (magnitudeImage.getVal(x+1,y) + magnitudeImage.getVal(x+1,y-1))/2;
                                pb = (magnitudeImage.getVal(x-1,y) + magnitudeImage.getVal(x-1,y+1))/2;
                        }
                        else if((orientDeg > 45 && orientDeg <= 90) || (orientDeg > -135 && orientDeg <= -90)) {
                                pa = (magnitudeImage.getVal(x+1,y-1) + magnitudeImage.getVal(x,y-1))/2;
                                pb = (magnitudeImage.getVal(x-1,y+1) + magnitudeImage.getVal(x,y+1))/2;
                        }
                        else if((orientDeg > 90 && orientDeg <= 135) || (orientDeg > -90 && orientDeg <= -45)) {
                                pa = (magnitudeImage.getVal(x,y-1) + magnitudeImage.getVal(x-1,y-1))/2;
                                pb = (magnitudeImage.getVal(x,y+1) + magnitudeImage.getVal(x+1,y+1))/2;
                        }
                        else if((orientDeg > 135 && orientDeg <= 180) || (orientDeg > -45 && orientDeg < 0)) {
                                pa = (magnitudeImage.getVal(x-1,y-1) + magnitudeImage.getVal(x-1,y))/2;
                                pb = (magnitudeImage.getVal(x+1,y+1) + magnitudeImage.getVal(x+1,y))/2;
                        }

                        if(mag > pa && mag > pb) {
                                if(mag < loThresh)
                                        continue;
                                else if(mag > hiThresh)
                                        edgeImage.setVal(x,y,255);
                                else if(traceContour(magnitudeImage,x,y, -1, 0)) {
                                        edgeImage.setVal(x,y,255);
                                }
                        }
                }
}

void V1::cannyEdgeDetect(const Image<byte> source, Image<byte> &edgeImage, Image<float> &orientationImage) {

        //Image<byte> edgeImage(source.getDims(), ZEROS);            //Image to return
        Image<float> magnitudeImage(source.getDims(), ZEROS);      //First Derivative Magnitude Image
        //Image<float> orientationImage(source.getDims(), ZEROS);    //First Derivative Orientation Image

        //gaussian blurred image -- to reduce noise
        Image<float> fblurImage = luminance(convGauss(source, sigma, sigma,10));

        //Find the magnitude and orientation
        gradient(fblurImage, magnitudeImage, orientationImage);

        //Perform non-maximul suppression and contour tracing to determine edges
        nonMaxSuppressAndContTrace(edgeImage, magnitudeImage, orientationImage);

}


#endif
