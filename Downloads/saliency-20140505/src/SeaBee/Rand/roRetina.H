/*
 *  roRetina.h
 *
 *
 *  Created by Randolph Voorhies on 11/13/07.
 *
 */
#ifndef RORETINA_H
#define RORETINA_H

#include <queue>
#include "pix.H"
#include "Image/PyrBuilder.H"
#include "Image/PyramidOps.H"
#include "rutz/shared_ptr.h"
#include "roUtil.H"

using namespace std;

//A rank order retina.
template <class T>
class roRetina {
public:
        roRetina();
        void update(rutz::shared_ptr< Image<T> > inputImage);
        rutz::shared_ptr <priority_queue <pix <T> > > getSpikeWave();
private:
        rutz::shared_ptr < GaussianPyrBuilder<T> >      itsPyrBuilder;  //A pyramid builder for computing the gaussian pyramids
        rutz::shared_ptr < ImageSet < T > >             itsIPyramid;        //Intensity pyramid
        rutz::shared_ptr < priority_queue < pix <T> > > itsSpikeWave;   //A priority queue representing a single spike wave

        rutz::shared_ptr < Image<T> >                   itsCenterSurroundImage;
};


template <class T>
roRetina<T>::roRetina() :
 itsPyrBuilder(new GaussianPyrBuilder<T>(5)),
 itsIPyramid(new ImageSet<T>(5)),
 itsSpikeWave(new priority_queue< pix <T> >),
 itsCenterSurroundImage(new Image<T>)
{

}

template <class T>
void roRetina<T>::update(rutz::shared_ptr< Image <T> > inputImage) {
 //Build the 9 level Gaussian Image Pyramid
 *itsIPyramid = itsPyrBuilder->build(*inputImage,0,8);

 //Compute the center surround
 *itsCenterSurroundImage = centerSurround(*itsIPyramid,0,1);

 //Generate the spike wave from the center surround image
 itsSpikeWave = makePriQueue<float>(itsCenterSurroundImage);
}



template <class T>
rutz::shared_ptr <priority_queue <pix <T> > > roRetina<T>::getSpikeWave() {
        return itsSpikeWave;
}


#endif
