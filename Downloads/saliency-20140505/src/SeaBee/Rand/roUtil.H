/*
 *  roUtil.H
 *
 *
 *  Created by Randolph Voorhies on 11/13/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef ROUTIL_H
#define ROUTIL_H

#include <queue>

using namespace std;


//This function constructs a priority queue of type pix out of an image
template <class T>
rutz::shared_ptr< priority_queue< pix<T> > > makePriQueue(rutz::shared_ptr< Image<T> > inputImage) {
        rutz::shared_ptr< priority_queue< pix<T> > > outputQueue(new priority_queue< pix<T> >);

        for(int x=0; x<inputImage->getWidth(); x++) {
                for(int y=0; y<inputImage->getHeight(); y++) {
                        outputQueue->push(pix<T>(x,y,inputImage->getVal(x,y)));
                }
        }

        return outputQueue;
}

template <class T>
void makeHeatmap(rutz::shared_ptr<priority_queue < pix <T> > > rankOrderPixels, rutz::shared_ptr< Image< PixRGB <float> > > heatmap) {
        pix <float> currPix;
        float h=1;
        float i=255.0;
        float j=255.0;

        heatmap->clear();

        while(!rankOrderPixels->empty() && rankOrderPixels->top().val > 10) {
                currPix = rankOrderPixels->top();
                rankOrderPixels->pop();
                heatmap->setVal(currPix.x, currPix.y, PixRGB<float>(i,j,0.0));
                h-=.0001;
                i=255.0*h;
                j=255*(1-h);
        }
}

#endif

