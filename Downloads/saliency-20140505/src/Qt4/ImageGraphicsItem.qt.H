

#ifndef IMAGEGRAPHICSITEM_H
#define IMAGEGRAPHICSITEM_H

#include <QtGui/qgraphicsitem.h>
#include <QtGui/qevent.h>
#include <QtGui/qwidget.h>

#include "Component/ModelManager.H"
#include "Component/ModelComponent.H"
#include "Media/FrameSeries.H"
#include "Image/Image.H"
#include "Image/PixelsTypes.H"

class ImageGraphicsItem : public QObject, public QGraphicsItem
{
        Q_OBJECT;

  public:
    ImageGraphicsItem();

                //! Returns the bounding rectangle of this object
    QRectF boundingRect() const;

                //! Paint this object to the screen
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    void setImage(QImage img);

    void setSize(QSize size);

    QRectF getRect();

  private:
    QImage itsImage;
    QSize  itsSize;
};

#endif //IMAGEGRAPHICSITEM_H

