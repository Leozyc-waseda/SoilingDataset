#ifndef ANNOTATIONOBJECT_QT_C
#define ANNOTATIONOBJECT_QT_C

#include <math.h>

#include "NeovisionII/NeoAnnotate/AnnotationObject.qt.H"
#include "Util/log.H"
#include "NeovisionII/NeoAnnotate/ObjectAnimation.qt.H"
#include "NeovisionII/NeoAnnotate/AnimationDelegate.qt.H"
#include "rutz/trace.h"


#undef GVX_TRACE_EXPR
#define GVX_TRACE_EXPR false

qreal distanceToLine(QLineF line, QPointF point);

int AnnotationObject::theirIdCount = 1;

//######################################################################
AnnotationObjectVertex::AnnotationObjectVertex(AnnotationObject *parent, int frameNum, FrameRange frameRange, QPointF initialPos) :
  QGraphicsItem(parent), itsVertexSize(-4, -4, 8, 8), itsParent(parent),
  itsAnimation(new ObjectAnimation(frameNum, frameRange, initialPos))
{
  //Tell Qt that this item is available for drag & drop, and to no
  //apply scaling transformations to it as we zoom
  setFlags(
      QGraphicsItem::ItemIsMovable    |
      QGraphicsItem::ItemIgnoresTransformations |
      QGraphicsItem::ItemSendsGeometryChanges
      );

  setPos(initialPos);

  //Tell Qt to accept hover events over this vertex
#if INVT_QT4_MINOR >= 4
  setAcceptHoverEvents(true);
#else
  setAcceptsHoverEvents(true);
#endif

  itsColor = itsParent->getColor();
  itsColor.setAlpha(128);

  itsBrush = QBrush(itsColor);
  itsPen   = QPen(QBrush(QColor(0, 0, 0)), 1);
  itsPen.setCosmetic(true);

  QGraphicsItem::setZValue(2);

}

void AnnotationObjectVertex::makeTransparent()
{
  itsBrush = QBrush();
  itsPen   = QPen(Qt::NoPen);
  QGraphicsItem::setZValue(2);
}

void AnnotationObjectVertex::makeOpaque()
{
  itsPen   = QPen(QBrush(QColor(0,0,0)), 1);
  itsPen.setCosmetic(true);
  itsBrush = QBrush(itsColor);
  QGraphicsItem::setZValue(4);
}

const QRectF AnnotationObjectVertex::getVertexRect(void) const
{
  return itsVertexSize;
}


//######################################################################
QRectF AnnotationObjectVertex::boundingRect() const
{
  QTransform vp_trans = this->scene()->views()[0]->viewportTransform();

  qreal sx = vp_trans.m11();
  qreal sy = vp_trans.m22();

  QRect bbox(itsVertexSize.x()*sx, itsVertexSize.y()*sy, itsVertexSize.width()*sx, itsVertexSize.height()*sy);

  return bbox;
}

//######################################################################
void AnnotationObjectVertex::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  painter->setClipRect( option->exposedRect );

  //Draw this vertex as an ellipse
  painter->setPen(itsPen);
  painter->setBrush(itsBrush);
  painter->drawEllipse(this->getVertexRect());
}

//######################################################################
QVariant AnnotationObjectVertex::itemChange(GraphicsItemChange change, const QVariant &value)
{
  //Redraw the parent when this vertex is moved so that the polygon outline
  //redraws on the fly
  if(change == QGraphicsItem::ItemPositionHasChanged)
    itsParent->update();

  return QGraphicsItem::itemChange(change, value);
}

//######################################################################
void AnnotationObjectVertex::hoverEnterEvent(QGraphicsSceneHoverEvent * event)
{

  if(itsPen.style() != Qt::NoPen)
  {
    //Make the vertex a bit more opaque when there is a mouseover
    itsColor.setAlpha(255);
    itsBrush = QBrush(itsColor);
    itsPen   = QPen(QBrush(QColor(128, 128, 128)), 3);
    itsPen.setCosmetic(true);
    update();
  }

}

//######################################################################
void AnnotationObjectVertex::setKeyframe(int frameNum, QPointF pos, bool visible)
{
  itsAnimation->setPosition(frameNum, pos, visible);
}

//######################################################################
void AnnotationObjectVertex::hoverLeaveEvent(QGraphicsSceneHoverEvent * event)
{

  if(itsPen.style() != Qt::NoPen)
  {
    //Make the vertex a bit more transparent when there is a mouseover
    itsColor.setAlpha(128);
    itsBrush = QBrush(itsColor);
    itsPen   = QPen(QBrush(QColor(0, 0, 0)), 1);
    itsPen.setCosmetic(true);
    update();
  }
}

//######################################################################
void AnnotationObjectVertex::mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
{
  itsParent->pubPrepareGeometryChange();
  int currFrame = itsParent->getCurrentFrameNum();

  //Add our current position as a keyframe in the animation if this
  //object was moved by the mouse
  if(this->pos() != itsAnimation->getFrameState(currFrame).pos)
  {
    itsAnimation->setPosition(currFrame, pos());
    emit(animationChanged());
  }

  //Pass the event down to the default event handler
  QGraphicsItem::mouseReleaseEvent(event);
}

//######################################################################
void AnnotationObjectVertex::mousePressEvent(QGraphicsSceneMouseEvent * event)
{
  itsParent->selectVertex(this);

  //Pass the event down to the default event handler
  QGraphicsItem::mousePressEvent(event);
}

//######################################################################
void AnnotationObjectVertex::frameChanged(int frameNum)
{
  //Store the current frame number
  itsCurrentFrame = frameNum;

  //Compute our interpolated position and visibility
  ObjectAnimation::FrameState state = itsAnimation->getFrameState(frameNum);

  //Set our position and visibility
  QGraphicsItem::setPos(state.pos);
  QGraphicsItem::setVisible(state.visible);

  update();
}

//######################################################################
void AnnotationObjectVertex::clearAnimation()
{
  itsAnimation->clear();
  emit(animationChanged());
}

////////////////////////////////////////////////////////////


//######################################################################
AnnotationObject::AnnotationObject(int frameNum, FrameRange frameRange, QPointF initialPos, QString description, int type, QGraphicsItem * parent) :
  QGraphicsItem(parent),
  itsId(theirIdCount++),
  itsDescription(description),
  itsType(type),
  itsAnimation(new ObjectAnimation(frameNum, frameRange, initialPos)),
  itsAnimationModel(this)
{
  //Assign a random color to this object
  QStringList possibleColors = QColor::colorNames();
  itsColor = QColor(possibleColors[qrand()%possibleColors.size()]);

  //Add a square of vertices for the initial shape
  this->addVertex(itsVertices.size(), new AnnotationObjectVertex(
        this, frameNum, itsAnimation->getFrameRange(), QPointF(-20, -20))
      );

  this->addVertex(itsVertices.size(), new AnnotationObjectVertex(
        this, frameNum, itsAnimation->getFrameRange(), QPointF(20, -20))
      );

  this->addVertex(itsVertices.size(), new AnnotationObjectVertex(
        this, frameNum, itsAnimation->getFrameRange(), QPointF(20, 20))
      );

  this->addVertex(itsVertices.size(), new AnnotationObjectVertex(
        this, frameNum, itsAnimation->getFrameRange(), QPointF(-20, 20))
      );

  //Create the initial polygon from these vertices
  recalculatePoly();

  //Tell Qt that this item is available for drag & drop
  setFlags(QGraphicsItem::ItemIsMovable);

  //Deselect this object
  showDeselected();

  //When our animation object changes states, we need to update ourselves
  QWidget::connect(itsAnimation, SIGNAL(animationChanged()), &itsAnimationModel, SLOT(animationChanged()));

  QGraphicsItem::setZValue(1);//AnnotationObjectZValue);
  itsCurrentFrame = frameNum;
}

//######################################################################
AnnotationObject::~AnnotationObject()
{
  //Destroy all of the child vertices when this object is destroyed
  for(int i=0; i<itsVertices.size(); i++)
    delete itsVertices[i];

  //Destroy our animation
  delete itsAnimation;
}

//######################################################################
void AnnotationObject::forceId(int id)
{
  itsId = id;
  if(itsId >= theirIdCount)
    theirIdCount = itsId+1;
}

//######################################################################
void AnnotationObject::addVertex(int index, AnnotationObjectVertex * v)
{
  itsAnimationModel.beginInsertRow(index+1);
  //Insert this vertex between the two vertices of the closest line
  itsVertices.insert(index, v);
  itsAnimationModel.endInsertRow();

  //When our animation object changes states, we need to update ourselves
  QWidget::connect(v->getAnimation(), SIGNAL(animationChanged()), &itsAnimationModel, SLOT(animationChanged()));

  itsAnimationModel.animationChanged();
}

//######################################################################
void AnnotationObject::setVertices(std::map<int, ObjectAnimation::FrameState> VertexFrames)
{
  //Delete any existing vertices
  itsAnimationModel.beginRemoveRowsPub(QModelIndex(), 0, itsVertices.size()); 
  for(int i=0; i<itsVertices.size(); i++)
  {
    delete itsVertices[i];
  }
  itsVertices.clear();
  itsAnimationModel.endRemoveRow();

  std::map<int, ObjectAnimation::FrameState>::iterator vIt;
  for(vIt=VertexFrames.begin(); vIt!=VertexFrames.end(); vIt++)
  {
    QPointF pos = vIt->second.pos;

    AnnotationObjectVertex *newVertex =
      new AnnotationObjectVertex(this, itsCurrentFrame, itsAnimation->getFrameRange(), pos);

    //Clear all keyframe from this vertices animation, because we don't yet know where
    //its first keyframe is.
    newVertex->clearAnimation();

    this->addVertex(vIt->first, newVertex);
  }
}

//######################################################################
void AnnotationObject::scaleEvent(float scale)
{
  prepareGeometryChange();
  QPointF ctr = getCenter();
  for(int i=0; i<itsVertices.size(); i++)
  {
    ObjectAnimation::FrameState fs = itsVertices[i]->getFrameState(itsCurrentFrame);
    // Update Point
    itsVertices[i]->setKeyframe(itsCurrentFrame,(fs.pos-ctr)*scale+ctr,fs.visible);
  }
  updateAnimation();
  //update();
}

//######################################################################
void AnnotationObject::expandPolygon()
{
  scaleEvent(2.0);
}

//######################################################################
void AnnotationObject::shrinkPolygon()
{
  scaleEvent(0.5);
}

//######################################################################
void AnnotationObject::stretchPolygon(QPointF dpos)
{
  prepareGeometryChange();
  QPointF ctr = getCenter();
  for(int i=0; i<itsVertices.size(); i++)
  {
    ObjectAnimation::FrameState fs = itsVertices[i]->getFrameState(itsCurrentFrame);
    // Update Point
    QPointF newPos;
    // Is this point left or right of center?
    if(fs.pos.x() < ctr.x()) // Vertex is on the left
      newPos.setX(fs.pos.x() - dpos.x());
    else
      newPos.setX(fs.pos.x() + dpos.x());
    // Is this point above or below of center?
    if(fs.pos.y() < ctr.y()) // Vertex is above
      newPos.setY(fs.pos.y() - dpos.y());
    else
      newPos.setY(fs.pos.y() + dpos.y());
    
    itsVertices[i]->setKeyframe(itsCurrentFrame,newPos,fs.visible);
  }
  updateAnimation();
  //update();
}


//######################################################################
void AnnotationObject::endTrack()
{
  // TODO: Must check that there is a next frame.  If there is not, there is no need to make the next keyframe invisible
  ObjectAnimation::FrameState fs = getFrameState(itsCurrentFrame);
  // If current frame is already invisible, no need to do anything
  if(fs.visible == false)
    return;

  setKeyframe(itsCurrentFrame+1,fs.pos,false);
  //updateAnimation();
}

//######################################################################
void AnnotationObject::insertVertexAtPoint(QPointF point)
{
  //Map from the scene coordinate system to our local coordinate system
  point = mapFromScene(point);

  //Find the distance from the clicked point to the first line
  QLineF line(itsVertices[0]->pos(), itsVertices[1]->pos());
  int minDistIdx = 1;
  qreal minDist  = distanceToLine(line, point);

  //Find the distance from the clicked point to the middle set of lines
  for(int i=2; i<itsVertices.size(); i++)
  {
    line = QLineF(itsVertices[i-1]->pos(), itsVertices[i]->pos());
    qreal dist = distanceToLine(line, point);
    if(dist < minDist)
    {
      minDist = dist;
      minDistIdx = i;
    }
  }

  //Find the distance from the clicked point to the last line
  line = QLineF(itsVertices[itsVertices.size()-1]->pos(), itsVertices[0]->pos());
  qreal dist = distanceToLine(line, point);
  if(dist < minDist)
  {
    minDist = dist;
    minDistIdx = 0;
  }

  //Create a new vertex, and drop it onto the clicked point
  AnnotationObjectVertex *newVertex =
    new AnnotationObjectVertex(this, itsCurrentFrame, itsAnimation->getFrameRange(), point);

  this->addVertex(minDistIdx, newVertex);
}

//######################################################################
void AnnotationObject::removeVertex(QPointF point)
{
  //Grab the pointer to the item at the clicked point
  QGraphicsItem * item = scene()->itemAt(point);

  //If there is no item there, then there's nothing to do
  if(item == NULL) return;

  //If we have only three vertices left, then we can't delete any
  if(itsVertices.size() <= 3) return;

  //Find out if the item has the same address as one of our vertices
  for(int i=0; i<itsVertices.size(); i++)
  {
    if( item == itsVertices[i] )
    {
      itsAnimationModel.beginRemoveRow(i+1);
      //If we have found our vertex, then we need to delete it.
      itsVertices.removeAt(i);
      delete item;
      itsAnimationModel.endRemoveRow();
      break;
    }
  }
}

//######################################################################
QPointF AnnotationObject::getCenter() const
{
  //From http://local.wasp.uwa.edu.au/~pbourke/geometry/polyarea/

  float A = 0;
  for(int i=0; i<itsVertices.size()-1; i++)
  {
    A +=
      (
       itsVertices[i]->pos().x() * itsVertices[i+1]->pos().y() -
       itsVertices[i+1]->pos().x() * itsVertices[i]->pos().y()
      );
  }
  A +=
    (
     itsVertices[itsVertices.size()-1]->pos().x() * itsVertices[0]->pos().y() -
     itsVertices[0]->pos().x() * itsVertices[itsVertices.size()-1]->pos().y()
    );

  A *= .5;

  float cx = 0;
  for(int i=0; i<itsVertices.size()-1; i++)
  {
    cx +=
      (itsVertices[i]->pos().x() + itsVertices[i+1]->pos().x()) *
      (
       itsVertices[i]->pos().x()*itsVertices[i+1]->pos().y() -
       itsVertices[i+1]->pos().x()*itsVertices[i]->pos().y()
      );
  }
  cx +=
    (itsVertices[itsVertices.size()-1]->pos().x() + itsVertices[0]->pos().x()) *
    (
     itsVertices[itsVertices.size()-1]->pos().x()*itsVertices[0]->pos().y() -
     itsVertices[0]->pos().x()*itsVertices[itsVertices.size()-1]->pos().y()
    );
  cx /= 6*A;

  float cy = 0;
  for(int i=0; i<itsVertices.size()-1; i++)
  {
    cy +=
      (itsVertices[i]->pos().y() + itsVertices[i+1]->pos().y()) *
      (
       itsVertices[i]->pos().x()*itsVertices[i+1]->pos().y() -
       itsVertices[i+1]->pos().x()*itsVertices[i]->pos().y()
      );
  }
  cy +=
    (itsVertices[itsVertices.size()-1]->pos().y() + itsVertices[0]->pos().y()) *
    (
     itsVertices[itsVertices.size()-1]->pos().x()*itsVertices[0]->pos().y() -
     itsVertices[0]->pos().x()*itsVertices[itsVertices.size()-1]->pos().y()
    );
  cy /= 6*A;

  return QPointF(cx, cy);
}

//######################################################################
QRectF AnnotationObject::boundingRect() const
{
  return itsPoly.boundingRect() | childrenBoundingRect();
}

//######################################################################
void AnnotationObject::recalculatePoly()
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  //Tell Qt that we are about to change the shape of this object

  //Construct our polygon from our vertices which are stored in clockwise order
  QPolygonF poly;
  for(int i=0; i<itsVertices.size(); i++)
  {
    if(itsVertices[i]->getFrameState(itsCurrentFrame).visible)
      poly.append(itsVertices[i]->pos());
  }
  itsPoly = poly;
}

//######################################################################
void AnnotationObject::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  //Recalculate the polygon on a paint, because it's likely we got here from an update() which was called
  //when something was changed
  recalculatePoly();
  painter->setClipRect( option->exposedRect );

  if(itsSelected)
  {
    //Make the polygon a bit more opaque when selected
    itsColor.setAlpha(128);
    itsBrush = QBrush(itsColor);
    itsPen = QPen(QBrush(QColor(255,255,255)), 2, Qt::SolidLine, Qt::RoundCap);
    itsPen.setCosmetic(true);
  }
  else
  {
    //Make the polygon a bit more transparent when deselected
    itsColor.setAlpha(int(float(itsOpacity)/100.0*255.0));
    itsBrush = QBrush(itsColor);
    itsPen   = QPen(Qt::NoPen);
  }

  //Draw this object as a colored polygon with a thick border
  painter->setPen(itsPen);
  painter->setBrush(itsBrush);
  painter->drawPolygon(itsPoly);
}

//######################################################################
void AnnotationObject::showSelected()
{
  itsSelected = true;

  for(int i=0; i<itsVertices.size(); i++)
    itsVertices[i]->makeOpaque();

  //Bring this object to the front so we can be sure to edit it
  QGraphicsItem::setZValue(3);

  update();
}

//######################################################################
void AnnotationObject::showDeselected()
{
  itsSelected = false;

  for(int i=0; i<itsVertices.size(); i++)
    itsVertices[i]->makeTransparent();

  QGraphicsItem::setZValue(1);
  update();
}

//######################################################################
void AnnotationObject::frameChanged(int frameNum)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  //Store the current frame number
  itsCurrentFrame = frameNum;

  //Compute our interpolated position and visibility
  ObjectAnimation::FrameState state = itsAnimation->getFrameState(frameNum);

  //Set our position and visibility
  QGraphicsItem::setPos(state.pos);
  QGraphicsItem::setVisible(state.visible);


  if(state.visible)
  {
    for(int i=0; i<itsVertices.size(); i++)
      itsVertices[i]->frameChanged(frameNum);
  }

  itsAnimationModel.animationChanged();

  update();
}

void AnnotationObject::setKeyframe(int frameNum, QPointF pos, bool visible)
{
  itsAnimation->setPosition(frameNum, pos, visible);
}

//######################################################################
void AnnotationObject::updateAnimation()
{
  frameChanged(itsCurrentFrame);
}

//######################################################################
void AnnotationObject::mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
{
  //Add our current position as a keyframe in the animation if this
  //object was moved by the mouse
  if(this->pos() != itsAnimation->getFrameState(itsCurrentFrame).pos)
  {
		for(int i=0; i<itsVertices.size(); ++i)
		{
			ObjectAnimation::FrameState vertState = itsVertices[i]->getFrameState(itsCurrentFrame);
			itsVertices[i]->setKeyframe(itsCurrentFrame, vertState.pos, vertState.visible);
		}

    itsAnimation->setPosition(itsCurrentFrame, pos());
    emit(animationChanged(0, itsCurrentFrame));
  }

  //Pass the event down to the default event handler
  QGraphicsItem::mouseReleaseEvent(event);
}

//######################################################################
void AnnotationObject::mousePressEvent(QGraphicsSceneMouseEvent * event)
{
  //Let everyone know that we were selected
  emit(objectSelected(itsId));

  //Let the animation view know that the main object was selected
  emit(rowSelected(0));

  itsAnimationModel.animationChanged();

  //Pass the event down to the default event handler
}


void AnnotationObject::constructAnimationContextMenu(QPoint globalPos, int row, int column)
{
  if(row == 0)
  {
    //Tell the animation object to construct a context menu, and perform the requisite action
    itsAnimation->constructContextMenu(globalPos, column);

    //Recompute our object's position with the current frame in case something was changed
    //in our animation
    frameChanged(itsCurrentFrame);
  }
  else
  {
    itsVertices[row-1]->getAnimation()->constructContextMenu(globalPos, column);
  }
  //Repaint ourselves
  update();
}


//######################################################################
AnimationModel* AnnotationObject::getAnimationModel()
{
  return &itsAnimationModel;
}


//######################################################################
void AnnotationObject::selectVertex(AnnotationObjectVertex * vertex)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  //Find which row contains the given vertex
  int row=-1;
  for(int i=0; i<itsVertices.size(); i++)
  {
    if(itsVertices[i] == vertex)
    {
      row = i;
      break;
    }
  }
  ASSERT(row >= 0);

  row = row+1;

  //Let everyone know that we were selected
  emit(objectSelected(itsId));

  //Let the animation view know that this particular vertex was selected
  emit(rowSelected(row));

  //Tell the model to tell the view to refresh the whole table
  itsAnimationModel.animationChanged();
}

//######################################################################
QPainterPath AnnotationObject::shape() const
{
  QPainterPath path;
  path.addPolygon(itsPoly);
  return path;
}

//######################################################################
QVector<QMap<int, QPointF> > AnnotationObject::renderAnimation()
{
  //Create an empty set of frame/position mappings for each of our vertices
  QVector<QMap<int, QPointF> > vertexAnimations(itsVertices.size());

  for(int fnum=itsAnimation->getFrameRange().getFirst(); fnum<itsAnimation->getFrameRange().getLast(); ++fnum)
  {
    ObjectAnimation::FrameState parentState = itsAnimation->getFrameState(fnum);
    if(parentState.visible)
    {
      //Loop through each vertex for this frame, and render out the position
      for(int vIdx=0; vIdx<itsVertices.size(); ++vIdx)
      {
        ObjectAnimation::FrameState vertexState = itsVertices[vIdx]->getFrameState(fnum);
        
        if(vertexState.visible)
        {
          vertexAnimations[vIdx][fnum] = vertexState.pos + parentState.pos;
        }
      }
    }
  }
  return vertexAnimations;
}

//######################################################################
std::map<int, ObjectAnimation::FrameState> AnnotationObject::getVertexStates(int fnum)
{
  std::map<int, ObjectAnimation::FrameState> vertexStates;

  ObjectAnimation::FrameState objState = this->itsAnimation->getFrameState(fnum);


  for(int vIdx=0; vIdx<itsVertices.size(); ++vIdx)
  {
    ObjectAnimation::FrameState vState = itsVertices[vIdx]->getFrameState(fnum);

    //Ensure that if the object is invisible, then all of its vertices are as well
    //if(!objState.visible)
    //  vState.visible = false;

    vState.pos = vState.pos + objState.pos;

    vertexStates[vIdx] = vState;
  }

  return vertexStates;
}

//######################################################################
void AnnotationObject::clearAnimation()
{
  itsAnimation->clear();
}



















//######################################################################
qreal distanceToLine(QLineF line, QPointF point)
{
  //From http://www.codeguru.com/forum/showthread.php?t=194400
  qreal distanceLine = 0;
  qreal distanceSegment = 0;

  double r_numerator = (point.x()-line.x1())*(line.x2()-line.x1()) + (point.y()-line.y1())*(line.y2()-line.y1());
  double r_denomenator = (line.x2()-line.x1())*(line.x2()-line.x1()) + (line.y2()-line.y1())*(line.y2()-line.y1());
  double r = r_numerator / r_denomenator;
  //
  //FIXME: UNUSED? double px = line.x1() + r*(line.x2()-line.x1());
  //double py = line.y1() + r*(line.y2()-line.y1());
  //
  double s =  ((line.y1()-point.y())*(line.x2()-line.x1())-(line.x1()-point.x())*(line.y2()-line.y1()) ) / r_denomenator;

  distanceLine = fabs(s)*sqrt(r_denomenator);

  //
  // (xx,yy) is the point on the lineSegment closest to (point.x(),point.y())
  //
  //FIXME: UNUSED? double xx = px;
  //double yy = py;

  if ( (r >= 0) && (r <= 1) )
  {
    distanceSegment = distanceLine;
  }
  else
  {


    double dist1 = (point.x()-line.x1())*(point.x()-line.x1()) + (point.y()-line.y1())*(point.y()-line.y1());
    double dist2 = (point.x()-line.x2())*(point.x()-line.x2()) + (point.y()-line.y2())*(point.y()-line.y2());
    if (dist1 < dist2)
    {
      //xx = line.x1();
      //yy = line.y1();
      distanceSegment = sqrt(dist1);
    }
    else
    {
      //xx = line.x2();
      //yy = line.y2();
      distanceSegment = sqrt(dist2);
    }


  }

  return distanceSegment;
}

#endif //ANNOTATIONOBJECT_QT_C
