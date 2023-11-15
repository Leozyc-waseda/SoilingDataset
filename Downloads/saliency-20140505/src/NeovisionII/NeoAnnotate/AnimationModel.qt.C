#ifndef ANIMATIONMODEL_QT_C
#define ANIMATIONMODEL_QT_C

#include "NeovisionII/NeoAnnotate/AnnotationObject.qt.H"
#include "NeovisionII/NeoAnnotate/ObjectAnimation.qt.H"
#include "NeovisionII/NeoAnnotate/AnimationDelegate.qt.H"


AnimationModel::AnimationModel(AnnotationObject *parent)
{
  itsParent   = parent;
  itsVertices = itsParent->getVertices();

  setSupportedDragActions(Qt::MoveAction);
}

QVariant AnimationModel::headerData(int section, Qt::Orientation orientation, int role) const
{
  if(role != Qt::DisplayRole)
    return QVariant();

  //Show a small header on the side of the animation view
  if(orientation == Qt::Vertical)
  {
    //The first row is always the object, and all subsequent rows are vertices
    if(section == 0)
      return QString("Object");

    return QString("Vertex");
  }

  return QString();
}

int AnimationModel::rowCount(const QModelIndex &parent) const
{
  return itsVertices->size()+1;
}

int AnimationModel::columnCount(const QModelIndex &parent) const
{
  FrameRange range = itsParent->getAnimation()->getFrameRange();
  int numFrames = range.getLast() - range.getFirst()+1;

  return numFrames;
}

QVariant AnimationModel::data(const QModelIndex &index, int role) const
{
  FrameRange range = itsParent->getAnimation()->getFrameRange();

  //Make sure we are trying to grab data from a valid table cell
  if(role != Qt::DisplayRole || !index.isValid())
    return QVariant();
  if(index.column() > range.getLast() - range.getFirst() || index.column() < 0)
    return QVariant();
  if(index.row() > itsVertices->size())
    return QVariant();

  int frameNum = index.column() - range.getFirst();

  //Grab the animation that is related to the requested row
  ObjectAnimation* animation;
  if(index.row() == 0)
    animation = itsParent->getAnimation();
  else
    animation = itsVertices->at(index.row()-1)->getAnimation();

  //Get the frametype from the animation
  return animation->getFrameType(frameNum);
}

bool AnimationModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
  return false;
}

Qt::ItemFlags AnimationModel::flags(const QModelIndex &index) const
{
  if(!index.isValid()) return 0;

  FrameRange range = itsParent->getAnimation()->getFrameRange();
  int frameNum = index.column() - range.getFirst();

  //Grab the animation that is related to the requested row
  ObjectAnimation* animation;
  if(index.row() == 0)
    animation = itsParent->getAnimation();
  else
    animation = itsVertices->at(index.row()-1)->getAnimation();

  //Get the frametype from the animation
  AnimationDelegate::FrameType type = animation->getFrameType(frameNum);

  //If the frame is a keyframe, make it drag-and-dropable
  if(type == AnimationDelegate::VisibleKeyframe || type == AnimationDelegate::InvisibleKeyframe)
    return Qt::ItemIsDragEnabled | Qt::ItemIsSelectable |  QAbstractTableModel::flags(index);

  //If the index doesn't have a keyframe, then let Qt know we can select it, and drop things on it
  return QAbstractTableModel::flags(index) | Qt::ItemIsSelectable | Qt::ItemIsDropEnabled;
}

Qt::DropActions AnimationModel::supportedDropActions() const
{
  return Qt::MoveAction;
}

bool AnimationModel::dropMimeData(const QMimeData * data,
    Qt::DropAction action,
    int row, int col,
    const QModelIndex & parent)
{
  //Grab the destination for the drag
  int dstRow    = parent.row();
  int dstColumn = parent.column();

  //If the destination is invalid, then don't allow the drop
  if(dstRow == -1) return false;

  //Get the source from the last reported mouse click
  //Unfortunately, Qt just sorts the source coordinates by their coordinates, so we
  //have no idea which of the possibly multiple items the user clicked on to drag,
  //and thus no hints as to how to shift the new items. To get around this, we just
  //record every mouse click on the view, and pass it downstream until it gets to this
  //animation model.
  int srcRow    = itsLastClick.row();
  int srcColumn = itsLastClick.column();

  //If we're trying to drop an animation state onto different object, then
  //don't allow the drop
  if(srcRow != dstRow)
    return false;

  int columnDelta = dstColumn - srcColumn;

  //Unpack the MIME data to retrieve the source coordinates of the drag
  QStringList formats = data->formats();
  QByteArray encodedData = data->data(formats[0]);
  QDataStream stream(&encodedData, QIODevice::ReadOnly);
  while(!stream.atEnd())
  {
    //Each r,c coordinate is the original position of an item to be dragged
    int r, c;
    QMap<int, QVariant> roles;
    stream >> r >> c >> roles;

    //Grab the animation that is related to the requested row
    ObjectAnimation* animation;
    if(r == 0)
      animation = itsParent->getAnimation();
    else
      animation = itsVertices->at(r-1)->getAnimation();

    int oldFrameNum = c + animation->getFrameRange().getFirst();
    int newFrameNum = oldFrameNum + columnDelta;

    //If the drag is actually to drag a keyframe to a new location, then execute
    //the move
    animation->moveKeyframe(oldFrameNum, newFrameNum);
  }
  return true;
}

void AnimationModel::animationChanged()
{
  //When any of the animations have changed, just update the whole table. We
  //can get more clever later if need be.

  FrameRange range = itsParent->getAnimation()->getFrameRange();
  

  QModelIndex tl = this->index(0, 0);
  QModelIndex br = this->index(itsVertices->size(), range.getLast()-range.getFirst());
  emit(QAbstractItemModel::dataChanged(tl, br));
}

void AnimationModel::beginInsertRow(int row)
{
  this->beginInsertRows(QModelIndex(), row, row);
}

void AnimationModel::endInsertRow()
{
  this->endInsertRows();
}

void AnimationModel::beginRemoveRow(int row)
{

  this->beginRemoveRows(QModelIndex(), row, row);
}

void AnimationModel::beginRemoveRowsPub(const QModelIndex & i, int first, int last)
{
  this->beginRemoveRows(i, first, last);
}

void AnimationModel::endRemoveRow()
{
  this->endRemoveRows();
}


#endif //ANIMATIONMODEL_QT_C


