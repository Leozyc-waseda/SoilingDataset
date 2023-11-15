#ifndef OBJECTANIMATION_QT_C
#define OBJECTANIMATION_QT_C

#include "NeovisionII/NeoAnnotate/ObjectAnimation.qt.H"
#include "rutz/trace.h"

//######################################################################
ObjectAnimation::ObjectAnimation(int frameNum, FrameRange masterFrameRange, QPointF initialPos) :
  itsMasterFrameRange(masterFrameRange)
{
  itsKeyFrames.clear();

  //Insert a single keyframe at the current frame number at the given initial position
  FrameState defaultKey;
  defaultKey.pos     = initialPos;
  defaultKey.visible = true;
//  itsKeyFrames[masterFrameRange.getFirst()] = defaultKey;
  itsKeyFrames[frameNum] = defaultKey;
}

//######################################################################
std::pair<QMap<int, ObjectAnimation::FrameState>::const_iterator,
          QMap<int, ObjectAnimation::FrameState>::const_iterator>
ObjectAnimation::getBoundingKeyframes(int fnum)
{

  GVX_TRACE(__PRETTY_FUNCTION__);

  //Create our bounding keyframes container, and fill it with bogus
  //values, in case we're asked to find unreasonable bounding frames
  std::pair<QMap<int, ObjectAnimation::FrameState>::const_iterator,
            QMap<int, ObjectAnimation::FrameState>::const_iterator> ret;
  ret.first  = itsKeyFrames.end();
  ret.second = itsKeyFrames.end();

  //If we're asked for unreasonable bounding frames, then just return the
  //bogus container
  if(fnum < itsKeyFrames.begin().key())
    return ret;
  if(fnum > (itsKeyFrames.end()-1).key())
    return ret;

  //Find the first keyframe which has a larger or equal value
  //to the frame number
  ret.second = itsKeyFrames.lowerBound(fnum);

  //Try to find the previous keyframe as long as our frameNum isn't a keyframe, or
  //the first frame in the animation.
  ret.first = ret.second;
  if(ret.second != itsKeyFrames.begin() && ret.second.key() != fnum)
    ret.first--;

  return ret;
}

//######################################################################
ObjectAnimation::FrameState ObjectAnimation::getFrameState(int frameNum)
{
  GVX_TRACE(__PRETTY_FUNCTION__);


  //If the requested frame is before the first keyframe, just return an invisible
  //framestate
  if(frameNum < itsKeyFrames.begin().key())
  {
    FrameState ret;
    ret.visible  = false;
    ret.is_keyframe = false;
    return ret;
  }

  //If we are beyond the final keyframe, then just return the position of the last known one
  //Note that this means that all frames after the last keyframe will inherit that last keyframe's
  //visibility state
  if(frameNum > (itsKeyFrames.end()-1).key())
  {
    FrameState ret  = (itsKeyFrames.end()-1).value();
    ret.is_keyframe = false;
    return ret;
  }

  //Find the bounding keyframes for the given frame
  std::pair<QMap<int, ObjectAnimation::FrameState>::const_iterator,
            QMap<int, ObjectAnimation::FrameState>::const_iterator> bounds =
              getBoundingKeyframes(frameNum);

  //If the bounding keyframes are the same, then no need to interpolate - we
  //already have a keyframe
  if(bounds.first == bounds.second) 
  {
    return bounds.first.value();
  }

  FrameState upperKey = bounds.second.value();
  FrameState lowerKey = bounds.first.value();
  QPointF lowerPos = lowerKey.pos;
  QPointF upperPos = upperKey.pos;

  //Interpolate between the two neighboring keyframes
  float alpha = 1.0;
  if(bounds.second.key() != bounds.first.key())
  {
    alpha = float(frameNum - bounds.first.key()) /
            float(bounds.second.key() - bounds.first.key());
  }
  QPointF intPos = upperPos*alpha + lowerPos*(1.0-alpha);

  FrameState ret;
  ret.pos         = intPos;
  ret.visible     = lowerKey.visible;
  ret.is_keyframe = false;

  return ret;
}

//######################################################################
AnimationDelegate::FrameType ObjectAnimation::getFrameType(int fnum)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  //If the requested frame is before the first keyframe, just return an invisible frametype
  if(fnum < itsKeyFrames.begin().key())
    return AnimationDelegate::Invisible;

  //If we are beyond the final keyframe, then look to our previous keyframe to see if we are invisible or not
  if(fnum > (itsKeyFrames.end()-1).key())
  {
    FrameState ret  = (itsKeyFrames.end()-1).value();
    if(ret.visible)
      return AnimationDelegate::Tween;
    else
      return AnimationDelegate::Invisible;
  }

  //Find the bounding keyframes for the given frame
  std::pair<QMap<int, ObjectAnimation::FrameState>::const_iterator,
            QMap<int, ObjectAnimation::FrameState>::const_iterator> bounds =
              getBoundingKeyframes(fnum);

  //If we have an actual keyframe:
  if(bounds.first.key() == bounds.second.key())
  {
    if(bounds.first.value().visible)
      return AnimationDelegate::VisibleKeyframe;
    else
      return AnimationDelegate::InvisibleKeyframe;
  }

  if(bounds.first.value().visible)
    return AnimationDelegate::Tween;

  return AnimationDelegate::Invisible;
}

//######################################################################
bool ObjectAnimation::moveKeyframe(int from_fnum, int to_fnum)
{
  if(itsKeyFrames.contains(from_fnum) && !itsKeyFrames.contains(to_fnum))
  {
    if(to_fnum >= itsMasterFrameRange.getFirst() && to_fnum <= itsMasterFrameRange.getLast())
    {
      itsKeyFrames[to_fnum] = itsKeyFrames[from_fnum];
      itsKeyFrames.remove(from_fnum);
      emit(animationChanged());
      return true;
    }
  }

  return false;
}

//######################################################################
void ObjectAnimation::setPosition(int frameNum, QPointF pos, bool visible)
{
  //Construct a new keyframe at the given framenumber with the given position
  FrameState key;
  key.pos         = pos;
  key.visible     = visible;
  key.is_keyframe = true;
  //Insert the keyframe into the data store
  itsKeyFrames[frameNum] = key;

  emit(animationChanged());
}

//######################################################################
void ObjectAnimation::constructContextMenu(QPoint pos, int column)
{
  //Compute the absolute frame number
  int frameNum = column + itsMasterFrameRange.getFirst();

  //Create a new drop down menu
  QMenu menu;

  QAction *createKeyframe = NULL;
  QAction *removeKeyframe = NULL;
  QAction *makeVisible    = NULL;
  QAction *makeInvisible  = NULL;

  //Find the relevant frame number
  QMap<int, FrameState>::iterator frameIt = itsKeyFrames.find(frameNum);

  if(frameIt != itsKeyFrames.end())
  {
    //Create some actions if the active frame is a keyframe

    FrameState state = frameIt.value();

    //Allow the user to make the current keyframe visible or invisible
    if(state.visible == true)
    {
      makeInvisible = new QAction(tr("Make &Invisible"), this);
      menu.addAction(makeInvisible);
    }
    else if(state.visible == false)
    {
      makeVisible = new QAction(tr("Make &Visible"), this);
      menu.addAction(makeVisible);
    }

    //Allow the user to delete the current keyframe
    removeKeyframe = new QAction(tr("&Delete Keyframe"), this);
    menu.addAction(removeKeyframe);
  }
  else
  {
    //If the active frame is not a keyframe, allow the user to make it one
    createKeyframe = new QAction(tr("&Create Keyframe"), this);
    menu.addAction(createKeyframe);
  }

  //Show the menu and retrieve the selected action
  QAction* menuAction = menu.exec(pos);

  if(menuAction != NULL)
  {
    //Perform the requested action

    if(menuAction == createKeyframe)
    {
      FrameState newState  = getFrameState(frameNum);
      newState.visible     = true;
      newState.is_keyframe = true;
      itsKeyFrames[frameNum] = newState;
      emit(animationChanged());
    }
    else if(menuAction == removeKeyframe)
    {
      itsKeyFrames.erase(frameIt);
      emit(animationChanged());
    }
    else if(menuAction == makeVisible)
    {
      frameIt->visible = true;
      emit(animationChanged());
    }
    else if(menuAction == makeInvisible)
    {
      frameIt->visible = false;
      emit(animationChanged());
    }
  }

  //Clean up the menu items
  if(createKeyframe != NULL) delete createKeyframe;
  if(removeKeyframe != NULL) delete removeKeyframe;
  if(makeVisible    != NULL) delete makeVisible;
  if(makeInvisible  != NULL) delete makeInvisible;
}

void ObjectAnimation::clear()
{
  itsKeyFrames.clear();
  emit(animationChanged());
}

#endif //OBJECTANIMATION_QT_C
