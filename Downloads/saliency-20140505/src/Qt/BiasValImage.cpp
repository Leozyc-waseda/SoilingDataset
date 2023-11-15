/*! @file Qt/BiasValImage.cpp widget for display submap and updating bias */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/BiasValImage.cpp $
// $Id: BiasValImage.cpp 10794 2009-02-08 06:21:09Z itti $


/****************************************************************************
 ** ui.h extension file, included from the uic-generated form implementation.
 **
 ** If you want to add, delete, or rename functions or slots, use
 ** Qt Designer to update this file, preserving your code.
 **
 ** You should not define a constructor or destructor in this file.
 ** Instead, write your code in functions called init() and destroy().
 ** These will automatically be called by the form's constructor and
 ** destructor.
 *****************************************************************************/

#include "BiasValImage.h"
#include "Image/ShapeOps.H"

BiasValImage::BiasValImage(ComplexChannel& cc, SingleChannel &sc, int submap, QWidget *parent):
   QWidget(parent, 0), itsCC(cc), itsSC(sc), itsSubmap(submap), itsShowRaw(true), itsResizeToSLevel(true){

      QVBoxLayout* vlayout = new QVBoxLayout(this);

      //add the imageDisp, label and the bias set/get spin box
      itsSubmapDisp = new ImageCanvas(this);
      //set the image
      Image<float> img;

      if (itsSubmap != -1 ) { //single submaps
        if (itsShowRaw)
        {
          img = sc.getRawCSmap(itsSubmap);
          if (itsResizeToSLevel)
          {
            Dims mapDims = sc.getSubmap(0).getDims();
            // resize submap to fixed scale if necessary:
            if (img.getWidth() > mapDims.w())
              img = downSize(img, mapDims);
            else if (img.getWidth() < mapDims.w())
              img = rescale(img, mapDims);
          }


        } else {
          img = sc.getSubmap(itsSubmap);
        }
      } else { //combined submaps
        img = sc.getOutput();
      }

      //inplaceNormalize(img, 0.0F, 255.0F);
      //Image<PixRGB<byte> > colImg = toRGB(img);
      itsSubmapDisp->setImage(img);
      vlayout->addWidget(itsSubmapDisp);

      QString txtMsg;
      if (itsSubmap != -1 ) //single submap
         txtMsg = QString("Submap %1 weight").arg(itsSubmap);
      else
         txtMsg = QString("Total weight");

      QHBoxLayout* hlayout = new QHBoxLayout();
      QLabel* label = new QLabel(txtMsg, this);
      hlayout->addWidget(label);

      itsBiasVal = new QSpinBox( this);
      if (itsSubmap != -1 ) { //single submaps
         unsigned int clev = 0, slev = 0;
         sc.getLevelSpec().indexToCS(itsSubmap, clev, slev);
         LFATAL("FIXME");
         //itsBiasVal->setValue(int(sc.getCoeff(clev, slev)));
      } else { //combined submaps
         itsBiasVal->setValue(int(cc.getSubchanTotalWeight(sc)));
      }


      hlayout->addWidget( itsBiasVal );

      vlayout->addLayout( hlayout );

      //add the signal and slot for spin box
      connect( itsBiasVal, SIGNAL( valueChanged(int) ),
            this, SLOT( updateBias(int) ) );
   }

void BiasValImage::updateBias(int val){


   //update the coeff and the display

   Image<float> img;
   if (itsSubmap != -1 ){ //submaps
     unsigned int clev = 0, slev = 0;
     itsSC.getLevelSpec().indexToCS(itsSubmap, clev, slev);
         LFATAL("FIXME");
         //     itsSC.setCoeff(clev, slev, double(val));
     if (itsShowRaw)
     {
       img = itsSC.getRawCSmap(itsSubmap);
       if (itsResizeToSLevel)
       {
         Dims mapDims = itsSC.getSubmap(0).getDims();
         // resize submap to fixed scale if necessary:
         if (img.getWidth() > mapDims.w())
           img = downSize(img, mapDims);
         else if (img.getWidth() < mapDims.w())
           img = rescale(img, mapDims);
       }


     } else {
       img = itsSC.getSubmap(itsSubmap);
     }
   } else { //combined submaps
     itsCC.setSubchanTotalWeight(itsSC, double(val));
     img = itsSC.getOutput();
   }

   //inplaceNormalize(img, 0.0F, 255.0F);
   //Image<PixRGB<byte> > colImg = toRGB(img);
   itsSubmapDisp->setImage(img);


   //signal the update of the single channel output
   //If we are the combined output, then dont signal
   if (itsSubmap != -1)
      emit updateOutput();
}


void BiasValImage::updateValues()
{

   //update the coeff and the display
   Image<float> img;
   if (itsSubmap != -1 ) { //submaps
      unsigned int clev = 0, slev = 0;
      itsSC.getLevelSpec().indexToCS(itsSubmap, clev, slev);
      LFATAL("FIXME");
      //      itsBiasVal->setValue(int(itsSC.getCoeff(clev, slev)));
      if (itsShowRaw)
      {
        img = itsSC.getRawCSmap(itsSubmap);
        if (itsResizeToSLevel)
        {
          Dims mapDims = itsSC.getSubmap(0).getDims();
          // resize submap to fixed scale if necessary:
          if (img.getWidth() > mapDims.w())
            img = downSize(img, mapDims);
          else if (img.getWidth() < mapDims.w())
            img = rescale(img, mapDims);
        }


      } else {
        img = itsSC.getSubmap(itsSubmap);
      }
   } else {
     itsBiasVal->setValue(int(itsCC.getSubchanTotalWeight(itsSC)));
     img = itsSC.getOutput();
   }


   //inplaceNormalize(img, 0.0F, 255.0F);
   // Image<PixRGB<byte> > colImg = toRGB(img);
   itsSubmapDisp->setImage(img);


   //signal the update of the single channel output
   //If we are the combined output, then dont signal
   if (itsSubmap != -1)
      emit updateOutput();


}


void BiasValImage::setShowRaw(bool val)
{
  itsShowRaw = val;
}

void BiasValImage::setResizeToSLevel(bool val)
{
  itsResizeToSLevel = val;
}

