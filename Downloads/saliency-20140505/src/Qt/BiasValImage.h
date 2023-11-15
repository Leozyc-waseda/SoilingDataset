/*! @file Qt/BiasValImage.h display submap and change bias */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/BiasValImage.h $
// $Id: BiasValImage.h 7071 2006-08-30 00:05:15Z rjpeters $


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

#ifndef BIASVALIMAGE_h
#define BIASVALIMAGE_h
#include <qwidget.h>
#include <qspinbox.h>
#include "Qt/ImageCanvas.h"
#include "Channels/ComplexChannel.H"
#include "Channels/SingleChannel.H"
#include "Image/MathOps.H"
#include "Image/ColorOps.H"

//Object to respond to bias updates
class BiasValImage : public QWidget {
   Q_OBJECT

public:
      BiasValImage(ComplexChannel& cc, SingleChannel &sc, int submap, QWidget *parent);
      ~BiasValImage() {}

signals:
      void updateOutput();

public slots:
      void updateValues();
      void setShowRaw(bool val);
      void setResizeToSLevel(bool val);

private slots:
      void updateBias(int val);

private:
      ComplexChannel& itsCC;     // the complex channel that owns itsSC
      SingleChannel &itsSC;      //the single channel pointer
      int itsSubmap;             // the current submap we are showing
      ImageCanvas *itsSubmapDisp;   //the display
      QSpinBox *itsBiasVal;       // the value of the bias
      bool itsShowRaw;             //show raw submaps?
      bool itsResizeToSLevel;    //resize to saliency map level?

};

#endif

