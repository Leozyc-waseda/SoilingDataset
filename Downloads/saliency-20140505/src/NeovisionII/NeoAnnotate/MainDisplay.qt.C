
#ifndef MAINDISPLAY_QT_C
#define MAINDISPLAY_QT_C


#include <QtGui/qboxlayout.h>

#include "NeovisionII/NeoAnnotate/MainDisplay.qt.H"

MainDisplay::MainDisplay(QWidget *parent) :
  QGraphicsView(parent)
{
  //Create a new graphics scene which will handle all of our objects
  itsScene = new QGraphicsScene;
  setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);

  //Create a new display widget to show our movie images
  itsImageDisplay = new ImageGraphicsItem;
  itsScene->addItem(itsImageDisplay);

  //Set itsScene as the main scene for this viewer
  setScene(itsScene);

  //Preload the cursors for our various mouse actions
  QPixmap editCursorBmp("src/NeovisionII/NeoAnnotate/icons/cursor-arrow.png");
  itsEditCursor = QCursor(editCursorBmp, 7, 1);

  QPixmap addCursorBmp("src/NeovisionII/NeoAnnotate/icons/cursor-add.png");
  itsAddCursor = QCursor(addCursorBmp, 7, 1);

  QPixmap remCursorBmp("src/NeovisionII/NeoAnnotate/icons/cursor-rem.png");
  itsRemCursor = QCursor(remCursorBmp, 7, 1);

  // QPixmap rotCursorBmp("src/NeovisionII/NeoAnnotate/icons/rotate-icon.jpg");
  // itsRotCursor = QCursor(rotCursorBmp, 7, 1);

  setCursor(itsEditCursor);
}

void MainDisplay::setImage(QImage img)
{
  itsImageDisplay->setImage(img);
  itsImageDisplay->update();
  setSceneRect(itsImageDisplay->getRect());
}

void MainDisplay::zoomIn()
{
  scale(1.3, 1.3);
}

void MainDisplay::zoomOut()
{
  scale(1.0/1.3, 1.0/1.3);
}

void MainDisplay::mousePressEvent(QMouseEvent * event)
{
  //Convert the raw mouse click onto the possibly zoomed
  //and panned scene coordinates
  QMouseEvent * originalEvent = new QMouseEvent(*event);
  QPointF sceneClick = QGraphicsView::mapToScene(event->pos());
  switch(itsActionMode)
  {
  case Edit:
    if(event->button()==Qt::RightButton)
    {
      itsActionMode = Stretch;
      itsStretchStartPos = event->pos();      
    }
    else
    {
      QGraphicsView::mousePressEvent(originalEvent);
    }
    break;
  case Add:
    emit(addVertex(sceneClick));
    break;
  case Remove:
    emit(removeVertex(sceneClick));
    break;
  case Rotate:
    break;
  case Stretch:
    std::cout << "Why are we here?" <<std::endl;
    break;
  }

}

void MainDisplay::mouseMoveEvent(QMouseEvent * event)
{
  if(itsActionMode == Stretch)
  {
    QPointF dpos = event->pos()-itsStretchStartPos;
    // Update size of object
    emit(stretchPolygon(dpos));
    // Update new starting point of stretch
    itsStretchStartPos = event->pos();
  }
  else
  {
    QMouseEvent * originalEvent = new QMouseEvent(*event);
    QGraphicsView::mouseMoveEvent(originalEvent);
  }
}

void MainDisplay::mouseReleaseEvent(QMouseEvent * event)
{
  if(event->button()==Qt::RightButton && itsActionMode==Stretch)
  {
    itsActionMode = Edit;
  }
  else
  {
    QMouseEvent * originalEvent = new QMouseEvent(*event);
    QGraphicsView::mouseReleaseEvent(originalEvent);
  }
}

void MainDisplay::addObject(AnnotationObject * object)
{
  itsScene->addItem(object);
}

void MainDisplay::setActionMode_Cursor()
{
  setCursor(itsEditCursor);
  itsActionMode = Edit;
}

void MainDisplay::setActionMode_AddVertex()
{
  setCursor(itsAddCursor);
  itsActionMode = Add;
}

void MainDisplay::setActionMode_RemVertex()
{
  setCursor(itsRemCursor);
  itsActionMode = Remove;
}

// void MainDisplay::setActionMode_Rotate()
// {
//   setCursor(itsRotCursor);
//   itsActionMode = Rotate;
// }

#endif //MAINDISPLAY_QT_C

