/*! @file Qt/ImageCanvas.h widget for deling with images */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/ImageCanvas.h $
// $Id: ImageCanvas.h 12962 2010-03-06 02:13:53Z irock $


#ifndef IMAGECANVAS
#define IMAGECANVAS

#include <qwidget.h>
#include <qmainwindow.h>
#include <qpen.h>
#include <qpoint.h>
#include <qpixmap.h>
#include <qstring.h>
#include <qpointarray.h>
#include <qlabel.h>
#include <qlayout.h>
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "QtUtil/ImageConvert.H"
#include <qpopupmenu.h>

class QMouseEvent;

class ImageCanvas : public QWidget
{
        Q_OBJECT
public:
        ImageCanvas(QWidget *parent = 0, const char *name = 0 );
        ~ImageCanvas();
        //int getWidth();
        //int getHeight();
        std::vector<Point2D<int> > itsPointsClicked;
signals:
        void mousePressed(int x, int y, int button);

public slots:
  void paintEvent(QPaintEvent*);
        //! Convert the image to QPixmap and display
        void setImage(Image<PixRGB<byte> > &img);
        //! Convert the image to QPixmap and display
        void setImage(Image<float> &img);
        void saveImg();

        Image<PixRGB<byte> > getImage();


protected:
        //! Respond to mouse events
        void mousePressEvent( QMouseEvent *e );

private:
//        QLabel *itsQLabel;
        Image<PixRGB<byte> > itsImgByte;
        Image<float> itsImgFloat;


        // Used to store the objects placed on ImageCanvas
        //vector<Image<PixRGB<byte> > > itsObjects;

};

#endif
