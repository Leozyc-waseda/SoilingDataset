#ifndef ANNOTATIONOBJECT_QT_H
#define ANNOTATIONOBJECT_QT_H

#include <QtGui/QtGui>
#include "NeovisionII/NeoAnnotate/AnimationModel.qt.H"
#include "NeovisionII/NeoAnnotate/ObjectAnimation.qt.H"


class AnnotationObject;
class FrameRange;
//class ObjectAnimation;

class AnnotationObjectVertex : public QObject, public QGraphicsItem
{
  Q_OBJECT

  public:
    //! Create a new vertex to be a part of a parent Annotation Object
    AnnotationObjectVertex(AnnotationObject *parent, int frameNum, FrameRange frameRange, QPointF initialPos);

    //! Get the bounding rectangle of this vertex
    QRectF boundingRect() const;

    //! Paint this vertex on the screen
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    ObjectAnimation * getAnimation() { return itsAnimation; }

    ObjectAnimation::FrameState getFrameState(int fnum) { return itsAnimation->getFrameState(fnum); }

    //Make the object's brush and pen opaque
    void makeOpaque();

    //Make the object's brush and pen transparent
    void makeTransparent();

    //! Set a new keyframe
    void setKeyframe(int frameNum, QPointF pos, bool visible);

    //! Clear all keypoints from the animation
    void clearAnimation();

  signals:
    void animationChanged();

  public slots:
    //! Recompute our position and visibility for the given frame and redraw
    void frameChanged(int);

  protected:
    //! Overloaded to redraw the parent whenever the vertex is moved
    QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    //! Highlight this vertex when the mouse moves over
    void hoverEnterEvent(QGraphicsSceneHoverEvent * event);

    //! Dehighlight this vertex when the mouse leaves
    void hoverLeaveEvent(QGraphicsSceneHoverEvent * event);

    //! Create a new keyframe with the new position of the object when moved by the mouse
    //! and select this object
    void mouseReleaseEvent(QGraphicsSceneMouseEvent * event);

    //! Select the object when the user clicks on it
    void mousePressEvent(QGraphicsSceneMouseEvent * event);

  private:
    const QRectF getVertexRect(void) const;

    const QRectF itsVertexSize;   //!< Denotes how large to draw vertices

    AnnotationObject * itsParent; //!< A reference to the parent object
                                  //!< of which this vertex is a part

    QPen   itsPen;                //!< The pen to outline this vertex

    QBrush itsBrush;              //!< The brush to fill this vertex

    QColor itsColor;              //!< The color of this vertex

    ObjectAnimation * itsAnimation;

    int itsCurrentFrame;
};


class AnnotationObject : public QObject, public QGraphicsItem
{
  Q_OBJECT

  public:
    //! Create a new Annotation Object with a default description and type
    AnnotationObject(int frameNum, FrameRange frameRange, QPointF initialPos,
        QString description="None", int type=0,
        QGraphicsItem * parent = 0);

    //! Destroy this Annotation Object, and destroy all of it's vertices
    ~AnnotationObject();

    //! Insert a new vertex into the store at the given index
    void addVertex(int index, AnnotationObjectVertex * v);

    //! Add a vertex to this object at the given point
    //* This function will find the nearest line on this object's polygon, and will
    //  insert the new vertex between the two endpoints of that line */
    void insertVertexAtPoint(QPointF point);

    //! Remove the top most vertex of this object at the given point
    void removeVertex(QPointF point);

    //! Scale a polygon
    void scaleEvent(float scale);

    //! Shrink the size of a polygon
    void shrinkPolygon();
    
    //! Expand the size of a polygon
    void expandPolygon();

    //! Stretch the polygon given an offset
    void stretchPolygon(QPointF dpos);

    //! Make the next frame for this annotation an invisible keypoint
    void endTrack();

    //! Highlight this object
    void showSelected();

    //! Dehighlight this object
    void showDeselected();

    //! Get the bounding rectangle of this object, given by the bounding rect of it's polygon
    QRectF boundingRect() const;

    //! Draw this object on the screen
    /*! The paint function goes clockwise through each of the vertices in itsVertices, and constructs
     *  a QPolygon from their positions. This polygon is then drawn on the screen using a fill brush
     *  with itsColor, and a thick white line pen.*/
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    //! Get the Unique ID of this object.
    /*! This ID is auto-incremented by a static counter for each created object. */
    int     getId()    const { return itsId;    }

    //! Get the user specified description of this object
    QString getDescription()  const { return itsDescription;  }

    //! Get the user specified type of this object
    int getType()  const { return itsType;  }

    //! Get the randomly assigned color of this object
    QColor  getColor() const { return itsColor; }

    //! Compute the center point of this object
    QPointF getCenter() const;

    //! Set the description of this object
    void setDescription(QString description) { itsDescription = description; setToolTip(description); update(); }

    //! Set the type of this object
    void setType(int type) { itsType = type; update(); }

    //! Get the interpolated frame state for a given frame
    ObjectAnimation::FrameState getFrameState(int fnum) { return itsAnimation->getFrameState(fnum); }

    //! Get a pointer to our member animation
    ObjectAnimation * getAnimation() { return itsAnimation; }

    //! Get a pointer to the vertices structure
    QList<AnnotationObjectVertex*>* getVertices() { return &itsVertices; }

    //! Get a pointer to the animation model
    AnimationModel* getAnimationModel();

    //! Ask our ObjectAnimation member to construct a context menu
    void constructAnimationContextMenu(QPoint globalPos, int row, int column);

    int getCurrentFrameNum() { return itsCurrentFrame; }

    //! Tell the animation view that a vertex has been selected
    /*! This will then highlight the proper row in the view so that users can
        have some feedback about what they're editing*/
    void selectVertex(AnnotationObjectVertex *vertex);

    //! Return the actual shape of the polygon for hit detection
    QPainterPath shape() const;

    //! Allow vertices to prepare the object for a geometry change
    /*! When vertices are moved, they need to tell their parent object
     *  that the bounding box has been modified so things can be drawn
     *  correclty. This is called at the end of a vertex' mouseReleaseEvent
     *  to save a bit on processing power. */
    void pubPrepareGeometryChange() { this->prepareGeometryChange(); }

    //! Store the index of the last click in the animation model - used for drag and drop
    void setLastClick(const QModelIndex & index) {itsAnimationModel.setLastClick(index);}

    //! Set the user selectable opacity for this object
    void setOpacity(int opacity) { itsOpacity = opacity; update(); }

    //! Render this object's keyframed/relative animation out to an absolute structure.
    /*! The returned result of this method is a vector, which contains an entry for each
     *  vertex of the object.  Each of these entries is a Map, which maps framenumbers to
     *  absolute vertex positions at that framenumber. Some vertices may not have entries for
     *  every frame because they were made invisible.*/
    QVector<QMap<int, QPointF > > renderAnimation();

    //! Get all of the interpolated vertex states for a given frame number
    std::map<int, ObjectAnimation::FrameState> getVertexStates(int fnum);

    //! Reset all of the objects vertices to these values
    void setVertices(std::map<int, ObjectAnimation::FrameState> VertexFrames);

    //! Set a new keyframe
    void setKeyframe(int frameNum, QPointF pos, bool visible);

    //! Force the id of this object - this should really only be done when loading from XML
    void forceId(int id);

    //! Get a vertex by it's index
    AnnotationObjectVertex* getVertexById(int id) { return itsVertices[id]; }

    //! Clear all keypoints from the animation
    void clearAnimation();
  signals:
    //! Let the manager know that we have been selected by the user
    void objectSelected(int);

    //! Let the view know which row has been selected
    void rowSelected(int);

    //! Let the model know that the animation state has changed
    void animationChanged(int row, int frame);


  public slots:
    //! Recompute our position and visibility for the given frame and redraw
    void frameChanged(int);

    //! Recompute our position and visibility because our animation was changed
    void updateAnimation();

  protected:
    //! Create a new keyframe with the new position of the object when moved by the mouse
    //! and select this object
    void mouseReleaseEvent(QGraphicsSceneMouseEvent * event);

    //! Select the object when the user clicks on it
    void mousePressEvent(QGraphicsSceneMouseEvent * event);

  private:
    //! Recalculate itsPoly based on the currently visible vertices
    void recalculatePoly();

    static int theirIdCount;       //!< Count of # objects created for unique ids

    int itsId;               //!< The unique id of this object

    QString itsDescription;               //!< The user assigned description of this object

    int itsType;                   //!< The user assigned type of this object

    QPolygonF itsPoly;             //!< The polygon generated from the list of vertices

    QBrush itsBrush;               //!< The brush for drawing this object

    QPen itsPen;                   //!< The pen for outlining this object

    QColor itsColor;               //!< The color of this object

    int itsCurrentFrame;           //!< The current frame number of the animation

    ObjectAnimation *itsAnimation; //!< The object containing all of our animation information

    int itsOpacity;                //!< The user setable opacity

    bool itsSelected;              //!< Whether or not this object is selected by the user

    AnimationModel itsAnimationModel;

    QList<AnnotationObjectVertex*> itsVertices; //!< The list of vertices
                                                /*!< Vertices in this list are
                                                     in strictly clockwise ordering*/
};

//! A simple struct to hold the state of an annotation object at a single frame
/*! This struct is just used to store detailed info about annotation objects 
 *  when rendering animations out to XML or wherever. An std::map is necessary
 *  for the VertexFrame storage, as it will ensure that the frames are stored
 *  in order of ID so that they are removed and inserted in the same order.*/
struct AnnotationObjectFrame
{
  QString  ObjectDescription;
  int      ObjectType;
  int      ObjectId;
  ObjectAnimation::FrameState ObjectFrameState;
  std::map<int, ObjectAnimation::FrameState> VertexFrames;
};


#endif //ANNOTATIONOBJECT_QT_H


