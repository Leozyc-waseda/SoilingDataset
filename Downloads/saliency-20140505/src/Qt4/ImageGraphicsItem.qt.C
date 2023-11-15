#ifndef IMAGEGRAPHICSITEM_C
#define IMAGEGRAPHICSITEM_C

#include <QtGui/qgraphicsscene.h>
#include <QtGui/qpainter.h>
#include <QtGui/qstyleoption.h>
#include <QtCore/qcoreevent.h>

#include "Qt4/ImageGraphicsItem.qt.H"
#include "QtUtil/ImageConvert4.H"
#include "Image/Image.H"


// ######################################################################
ImageGraphicsItem::ImageGraphicsItem()
{
  QGraphicsItem::setZValue(0);
}

// ######################################################################
QRectF ImageGraphicsItem::boundingRect() const
{
  //return QRectF(0, 0, itsSize.width(), itsSize.height());
  return itsImage.rect();
}

// ######################################################################
void ImageGraphicsItem::setImage(QImage img)
{
  itsImage = img;
}

// ######################################################################
void ImageGraphicsItem::setSize(QSize size)
{
  itsSize = size;
  update();
}


// ######################################################################
void ImageGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
  painter->setClipRect( option->exposedRect );
  painter->drawImage(QPoint(0,0),itsImage);
}

QRectF ImageGraphicsItem::getRect()
{
  return itsImage.rect();
}
#endif //IMAGEGRAPHICSITEM_C

