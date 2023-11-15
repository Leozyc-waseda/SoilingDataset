#ifndef MAINDISPLAY_QT_H
#define MAINDISPLAY_QT_H

#include <QtGui/qwidget.h>
#include <QtGui/qgraphicsscene.h>
#include <QtGui/qgraphicsview.h>

//#include "NeovisionII/NeoAnnotate/ImageGraphicsItem.qt.H"
#include "Qt4/ImageGraphicsItem.qt.H"
#include "NeovisionII/NeoAnnotate/AnnotationObject.qt.H"
#include "QtUtil/ImageConvert4.H"

//MainDisplay is the widget which is responsible for drawing the main image as
//well as the handling the drawing, moving, etc. of the object outlines
class MainDisplay : public QGraphicsView
{
  Q_OBJECT

  public:
    //! Construct the Main Display
    /*! Set all sorts of Qt related parameters, and preload the cursor icons */
    MainDisplay(QWidget *parent = 0);

    //! Set the background image
    void setImage(QImage img);

    //! Add a new annotation object to be displayed/maniuplated
    void addObject(AnnotationObject * object);

    //! The three ways to manipulate annotation objects
    /*! Which of these is the current itsActionMode will determine
     *  how mouse events are handled, and whether they are passed
     *  through to the Qt subsystem*/
    enum ActionMode {Edit, Add, Remove, Rotate, Stretch};
  public slots:

    //! Scale our Graphics View up
    void zoomIn();

    //! Scale our Graphics View down
    void zoomOut();

    //! Set the current action mode to 'cursor' and change the mouse icon
    void setActionMode_Cursor();

    //! Set the current action mode to 'add vertex' and change the mouse icon
    void setActionMode_AddVertex();

    //! Set the current action mode to 'remove vertex' and change the mouse icon
    void setActionMode_RemVertex();

    // //! Set the current action mode to 'rotate' and change the mouse icon
    // void setActionMode_Rotate();

  signals:
    //! Emit a signal telling the Object Manager to tell the appropriate object to add a vertex
    void addVertex(QPointF);

    //! Emit a signal telling the Object Manager to tell the appropriate object to remove a vertex
    void removeVertex(QPointF);

    //! Emit a signal telling the Object Manager to tell the appropriate object to stretch
    void stretchPolygon(QPointF);

  protected:

    //! Perform an action based on the current action mode
    void mousePressEvent(QMouseEvent * event);

    //! 
    void mouseMoveEvent(QMouseEvent * event);

    void mouseReleaseEvent(QMouseEvent * event);

  private:
    //! The Graphics Item which is responsible for drawing the background image
    ImageGraphicsItem *itsImageDisplay;

    //! This Graphics Scene holds the background image, and all annotation objects
    QGraphicsScene *itsScene;

    //! The current action mode - see mousePressEvent()
    ActionMode itsActionMode;

    //! A default cursor icon
    QCursor itsEditCursor;
    QCursor itsAddCursor;
    QCursor itsRemCursor;
    //QCursor itsRotCursor;
    QPointF itsStretchStartPos;
};

#endif //MAINDISPLAY_QT_H

