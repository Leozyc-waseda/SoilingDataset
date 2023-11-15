/*!@file Corba/ImageOrbUtil.H conversion utils for Images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/ImageOrbUtil.H $
// $Id: ImageOrbUtil.H 6593 2006-05-16 20:33:52Z rjpeters $
//

//! Convert from various images structuers to send over CORBA

#ifndef ImageOrbUtil_H
#define ImageOrbUtil_H

#include "Corba/ImageOrbSK.hh"
#include "Image/Image.H"

//! image2Orb, convert and image object into a corba data type
template <class T> inline
ImageOrb* image2Orb(const Image<T>& img){

        ASSERT(img.initialized());
        int size = img.getSize()*sizeof(T);

        ImageOrb *imgOrb = new ImageOrb;

        imgOrb->width = img.getWidth();
        imgOrb->height = img.getHeight();
        imgOrb->pix_size = sizeof(T);                //the pixel size for error checking

        CORBA::Octet *dat = new CORBA::Octet[size];

        unsigned char *imgdata = const_cast<unsigned char*>
                                (reinterpret_cast<const unsigned char*>(img.getArrayPtr()));

        memcpy(dat, imgdata, size);

        imgOrb->data.replace(size, size, dat, 1);

        return imgOrb;
}

//! orb2Image, convert from a corba data type to and Image object
//! return the image in the function so we can find out the type we need
template <class T> inline
void orb2Image(const ImageOrb &imgOrb, Image<T>& target) {

        ASSERT(sizeof(T) == imgOrb.pix_size);        //check is the pixel size matches

        Image<T> img;

        img.attach((T*)imgOrb.data.get_buffer(),
                                        imgOrb.width, imgOrb.height);

        target = img.deepcopy();
}



#endif

