/*! @file Qt/ImageCanvas.cpp widget for deling with images */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/ImageCanvas.cpp $
// $Id: ImageCanvas.cpp 12962 2010-03-06 02:13:53Z irock $


#include "ImageCanvas.h"
#include "Raster/Raster.H"
#include "Image/ShapeOps.H"
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include <qfiledialog.h>
#include <qpainter.h>

ImageCanvas::ImageCanvas( QWidget *parent, const char *name )
        : QWidget( parent, name )
{
        QHBoxLayout *layout = new QHBoxLayout( this );
        layout->setMargin( 0 );
        setBackgroundMode( QWidget::NoBackground );
        setCursor( Qt::crossCursor );
        setUpdatesEnabled(true);
}

ImageCanvas::~ImageCanvas()
{
}

//! convert the image to QPixmap and display
void ImageCanvas::setImage(Image<PixRGB<byte> > &img){

  //Store a copy of the new image locally
  itsImgByte = img;

  //Call the update function, which will in turn call paintEvent
  update();

}

void ImageCanvas::paintEvent(QPaintEvent *)
{
  QPainter painter(this);

  if(itsImgByte.initialized())
  {
    painter.drawPixmap(0,0, convertToQPixmap(
          rescale(itsImgByte, Dims(this->width(), this->height()))
          )
        );
  }
  else
  {
    painter.drawPixmap(0,0, convertToQPixmap(
          Image<PixRGB<byte> >(this->width(), this->height(), ZEROS)
          )
        );
  }

}

//int ImageCanvas::getWidth()
//{
//  ret
//}
//int ImageCanvas::getHeight();

void ImageCanvas::setImage(Image<float> &img){
  itsImgFloat = img;

  inplaceNormalize(img, 0.0F, 255.0F);
  itsImgByte = toRGB(img);

  //Call the update function, which will in turn call paintEvent
  update();
}

void ImageCanvas::saveImg(){
  static QString prevFilename = QString::null;
  static int imgNum = 0;

    if (itsImgByte.initialized())
    {
      QString file = QFileDialog::getSaveFileName( prevFilename,
          "PPM image (*.ppm)",
          this, "SaveImageDialog",
          "Save Image as..." );
      prevFilename = file;
      file.replace("%d", QString("%L1").arg(imgNum++));

      if (!file.isEmpty())
        Raster::WriteRGB(itsImgByte, file.ascii());
    }
    else
      LINFO("Image not initialized");
}

//TODO: Check out the pixel info functionality, it doesn't seem to be working correctly
void ImageCanvas::mousePressEvent( QMouseEvent *e )
{
  itsPointsClicked.push_back(Point2D<int>(e->x(),e->y()));
  LINFO("%d,%d",e->x(),e->y());
}
/*void ImageCanvas::mousePressEvent( QMouseEvent *e ){

  //get the position in the img coordinates
  const int newx = int((float(e->x())) *
      float(itsImgByte.getWidth()) / float(this->width()));

  const int newy = int((float(e->y())) *
      float(itsImgByte.getHeight()) / float(this->height()));

  //Show the pixel value if the right button is clicked
  if (e->button() == Qt::RightButton){
       //LINFO("Pix value at (%i,%i): (%i,%i,%i)", newx, newy,
       //pixVal[0], pixVal[1], pixVal[2]);
       //Raster::WriteRGB(itsImg, "out.ppm");


    static QPopupMenu *popupMenu = NULL;

    if (itsImgByte.initialized())
    {
      if (popupMenu) delete popupMenu;
      popupMenu = new QPopupMenu(this);
      CHECK_PTR(popupMenu);
      popupMenu->insertItem("Pixel Info");
      popupMenu->insertItem(QString("  Loc: (%L1,%L2)").arg(newx).arg(newy));

      if (itsImgFloat.initialized())
      {
        float pixVal = itsImgFloat.getVal(newx, newy);
        popupMenu->insertItem(QString("  Val :(%L1)").arg(pixVal));
      }
      else
      {
        PixRGB<byte> pixVal = itsImgByte.getVal(newx, newy);
        popupMenu->insertItem(QString("  Val :(%L1,%L2, %L3)").
            arg(pixVal[0]).arg(pixVal[1]).arg(pixVal[2]));
      }

      popupMenu->insertSeparator();


      popupMenu->insertItem("Save Image", this, SLOT(saveImg()));
      popupMenu->popup(mapToGlobal(e->pos()));
    }

  } else {
   // LINFO("Button is not a right mouse click");
    emit mousePressed(newx, newy, e->button());
  }
}*/

Image<PixRGB<byte> > ImageCanvas::getImage(){
        return itsImgByte;
}
