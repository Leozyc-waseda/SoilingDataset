#ifndef ANNOTATIONOBJECTMANAGER_C
#define ANNOTATIONOBJECTMANAGER_C

#include "NeovisionII/NeoAnnotate/AnnotationObjectManager.qt.H"
#include "Util/log.H"
#include "Util/Assert.H"

//######################################################################
AnnotationObjectManager::AnnotationObjectManager(QObject * parent) :
  QAbstractTableModel(parent)
{
  itsCurrentSelectedRow = -1;
}

void AnnotationObjectManager::setAnimationView(QTableView *animationView)
{
  itsAnimationView = animationView;
}

//######################################################################
int AnnotationObjectManager::rowCount(const QModelIndex &parent) const
{
  return itsObjects.size();
}

//######################################################################
int AnnotationObjectManager::columnCount(const QModelIndex &parent) const
{
  return 3;
}

//######################################################################
QVariant AnnotationObjectManager::data(const QModelIndex &index, int role) const
{

  //Make sure we are trying to grab data from a valid table cell
  if(role != Qt::DisplayRole || !index.isValid())
    return QVariant();
  if(index.row() >= itsObjects.size() || index.row() < 0)
    return QVariant();

  //Make sure we actually have data to grab
  if(itsObjects.size() <= 0) return QVariant();

  //Get a reference to the object in question based on the row
  AnnotationObject* obj = itsObjects[index.row()];

  //Grab the proper data from the object based on the indicated column
  QVariant ret;
  switch(index.column())
  {
    case 0:
      ret = obj->getId();
      break;
    case 1:
      ret = obj->getDescription();
      break;
    case 2:
      ret = obj->getType();
      break;
  }

  return ret;
}

//######################################################################
QVariant AnnotationObjectManager::headerData(int section, Qt::Orientation orientation,
    int role) const
{
  if(role != Qt::DisplayRole)
    return QVariant();

  if(orientation == Qt::Horizontal)
    switch(section)
    {
      case 0:
        return QVariant("ID");
      case 1:
        return QVariant("Description");
      case 2:
        return QVariant("Type");
      default:
        return QVariant("");

    }
  else
    return QString();
}

//######################################################################
bool AnnotationObjectManager::addObject(AnnotationObject * object)
{
  //Connect the new object's 'hovered' signal to our own 'itemHovered' slot
  connect(object, SIGNAL(objectSelected(int)), this, SLOT(objectSelected(int)));

  connect(object, SIGNAL(rowSelected(int)), this, SLOT(selectAnimationRow(int)));

  //Let Qt know that we are messing with the rows, and insert the object
  //into our object list
  beginInsertRows(QModelIndex(), rowCount(), rowCount());
  itsObjects.insert(rowCount(), object);
  endInsertRows();

  object->setOpacity(itsOpacity);


  selectObject(itsObjects.size()-1);

  return true;
}

//######################################################################
bool AnnotationObjectManager::setData(const QModelIndex &index, const QVariant &value, int role)
{
  //Make sure we're trying to edit a valid table cell
  if(!index.isValid() || role != Qt::EditRole)
  {
    return false;
  }

  //If the user didn't enter anything, then let's ignore this to avoid
  //having the user accidently clear out useful data
  if(value.toString() == "")
    return false;

  //Grab a reference to the object that is being edited
  AnnotationObject* obj = itsObjects[index.row()];

  //Edit the object's data based on the column
  switch(index.column())
  {
    case 0:
      return false;
    case 1:
      obj->setDescription(value.toString());
      break;
    case 2:
      obj->setType(value.toInt());
      break;
    default:
      return false;
  }

  //Let Qt know that we changed some data
  emit(dataChanged(index, index));
  return true;
}

//######################################################################
Qt::ItemFlags AnnotationObjectManager::flags(const QModelIndex &index) const
{
  //The first column (the id) is selectable, but not editable
  if(index.column() == 0)
    return QAbstractTableModel::flags(index) | Qt::ItemIsSelectable;

  //All other columns are editable
  return QAbstractTableModel::flags(index) | Qt::ItemIsEditable;
}


//######################################################################
void AnnotationObjectManager::constructAnimationContextMenu(QPoint globalPos, int row, int column)
{
  if(itsCurrentSelectedRow != -1)
    itsObjects[itsCurrentSelectedRow]->constructAnimationContextMenu(globalPos, row, column);

}

//######################################################################
void AnnotationObjectManager::select(const QModelIndex & index)
{
  selectObject(index.row());
}

//######################################################################
void AnnotationObjectManager::objectSelected(int itemId)
{
  //Linearly search the items to find which one is being hovered...
  //We can do something more efficient later on if need be.
  int row = -1;
  for(int i=0; i<itsObjects.size(); i++)
  {
    if(itsObjects[i]->getId() == itemId)
      row = i;
  }
  selectObject(row);
}


void AnnotationObjectManager::deselectObject(int rowIdx)
{
  itsObjects[rowIdx]->showDeselected();
}

void AnnotationObjectManager::selectObject(int rowIdx)
{
  //Make sure the index is ok
  if(rowIdx < 0 || rowIdx >= itsObjects.size())
  {
    LERROR("Bad ItemIDX: %d", rowIdx);
    return;
  }


  if(itsCurrentSelectedRow != -1)
  {
   deselectObject(itsCurrentSelectedRow);
  }

  //Inform the new object that it is selected
  itsObjects[rowIdx]->showSelected();

  //Keep track of which row was selected
  itsCurrentSelectedRow = rowIdx;

  //ObjectAnimation* animation = itsObjects[rowIdx]->getAnimation();
  itsAnimationView->setModel(itsObjects[rowIdx]->getAnimationModel());

  //If there's nothing to do, don't emit the signal to avoid an infinite loop
  if(rowIdx != itsCurrentSelectedRow)
  {
    emit(selectingObject(rowIdx));
  }
}

void AnnotationObjectManager::selectAnimationRow(int rowIdx)
{
  AnimationDelegate * del = static_cast<AnimationDelegate*>(itsAnimationView->itemDelegate());
  if(del != NULL)
    del->setSelectedRow(rowIdx);
  else
    LINFO("WARNING: Static Cast Returning NULL Pointer! (Line %d)", __LINE__);

  AnimationModel * mod = static_cast<AnimationModel*>(itsAnimationView->model());
  if(mod != NULL)
    mod->animationChanged();
  else
    LINFO("WARNING: Static Cast Returning NULL Pointer! (Line %d)", __LINE__);
}

void AnnotationObjectManager::setLastAnimViewClick(const QModelIndex & pos)
{
  selectAnimationRow(pos.row());

  if(itsCurrentSelectedRow != -1)
  {
    itsObjects[itsCurrentSelectedRow]->setLastClick(pos);
  }


}

//######################################################################
void AnnotationObjectManager::addVertex(QPointF point)
{
  if(itsCurrentSelectedRow == -1)
    return;

  itsObjects[itsCurrentSelectedRow]->insertVertexAtPoint(point);
}

//######################################################################
void AnnotationObjectManager::removeVertex(QPointF point)
{
  if(itsCurrentSelectedRow == -1)
    return;

  itsObjects[itsCurrentSelectedRow]->removeVertex(point);
}

//######################################################################
void AnnotationObjectManager::expandPolygon()
{
  if(itsCurrentSelectedRow == -1)
    return;

  itsObjects[itsCurrentSelectedRow]->expandPolygon();
}

//######################################################################
void AnnotationObjectManager::shrinkPolygon()
{
  if(itsCurrentSelectedRow == -1)
    return;

  itsObjects[itsCurrentSelectedRow]->shrinkPolygon();
}

void AnnotationObjectManager::endTrack()
{
  if(itsCurrentSelectedRow == -1)
    return;
  itsObjects[itsCurrentSelectedRow]->endTrack();
}

void AnnotationObjectManager::stretchPolygon(QPointF dpos)
{
  if(itsCurrentSelectedRow == -1)
    return;
  itsObjects[itsCurrentSelectedRow]->stretchPolygon(dpos);  
}

//######################################################################
void AnnotationObjectManager::removeObject()
{

  //Make sure there is a valid object to delete
  if(itsObjects.size() <= 0)      return;
  if(itsCurrentSelectedRow == -1) return;

  //Let Qt know we are about to start messing with the row structure
  beginRemoveRows(QModelIndex(), itsCurrentSelectedRow, itsCurrentSelectedRow);

  //Take the currently selected annotation object out of the object list,
  //and destroy it
  AnnotationObject * object = itsObjects.takeAt(itsCurrentSelectedRow);
  delete object;

  //If we're out of objects, then make sure to invalidate our current selected row
  if(itsObjects.size() == 0)
  {
    itsCurrentSelectedRow = -1;
  }
  else if(itsCurrentSelectedRow >= itsObjects.size())
  {
    //If we had selected the last row, we now need to select one less
    --itsCurrentSelectedRow;
  }

  //Let Qt know we're done messing with the row structure
  endRemoveRows();
}

//######################################################################
void AnnotationObjectManager::frameChanged(int fnum)
{
  itsCurrentFrame = fnum;
}

//######################################################################
void AnnotationObjectManager::setOpacity(int opacity)
{
  itsOpacity = opacity;
  for(int i=0; i<itsObjects.size(); ++i)
  {
    itsObjects[i]->setOpacity(opacity);
  }
}

//######################################################################
std::map<int, std::map<int,AnnotationObjectFrame> > 
AnnotationObjectManager::renderAnimations()
{
  if(itsObjects.size() == 0)
    return std::map<int, std::map<int, AnnotationObjectFrame> >();

  std::map<int, std::map<int, AnnotationObjectFrame > > frames;

  FrameRange range = itsObjects[0]->getAnimation()->getFrameRange();
  for(int frameNum = range.getFirst(); frameNum < range.getLast(); frameNum++)
  {
    for(int objIdx=0; objIdx<itsObjects.size(); objIdx++)
    {
      AnnotationObjectFrame f;
      //Get the state of this object at the current frame
      f.VertexFrames      = itsObjects[objIdx]->getVertexStates(frameNum);
      f.ObjectDescription = itsObjects[objIdx]->getDescription();
      f.ObjectType        = itsObjects[objIdx]->getType();
      f.ObjectId          = itsObjects[objIdx]->getId();
      f.ObjectFrameState  = itsObjects[objIdx]->getFrameState(frameNum);

      frames[frameNum][f.ObjectId] = f;
    }
  }

  return frames;
}

//######################################################################
void AnnotationObjectManager::clear()
{
  if(itsObjects.size() > 0)
  {
    deselectObject(itsCurrentSelectedRow);

    while(itsObjects.size() > 0)
      removeObject();
  }
  itsCurrentSelectedRow = -1;
  //itsAnimationView = NULL;
}

//######################################################################
AnnotationObject* AnnotationObjectManager::getObjectById(int id)
{
  for(int i=0; i< itsObjects.size(); i++)
  {
    if(itsObjects[i]->getId() == id)
      return itsObjects[i];
  }
  return NULL;
}




#endif //ANNOTATIONOBJECTMANAGER_C
