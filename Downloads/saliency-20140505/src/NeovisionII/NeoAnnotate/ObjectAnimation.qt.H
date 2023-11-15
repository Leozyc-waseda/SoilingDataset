#ifndef OBJECTANIMATION_QT_H
#define OBJECTANIMATION_QT_H

#include <QtGui/QtGui>
#include "Media/FrameRange.H"
#include "Util/log.H"
#include <utility>

#include "NeovisionII/NeoAnnotate/AnimationDelegate.qt.H"

class ObjectAnimation : public QObject
{
  Q_OBJECT

  public:

    //! A structure to hold all necessary information about frame
    struct FrameState
    {
      QPointF pos;
      bool    visible;
      bool    is_keyframe;
    };

    //! Construct a new object animation with a specified framerange and initial position.
    /*! The masterFrameRange parameter must be the FrameRange for the entire move so that
        frames can be properly mapped into the animation.*/
    ObjectAnimation(int frameNum, FrameRange masterFrameRange, QPointF initialPos);

    //! Perform position interpolation on the fly
    FrameState getFrameState(int frameNum);

    //! Move a keyframe from one frame to another (likely because of a drag-and-drop operation)
    bool moveKeyframe(int from_fnum, int to_fnum);

    //! Insert a new keyframe at frameNumber indicating the object is at position pos
    void setPosition(int frameNum, QPointF pos, bool visible=true);

    //! Draw a dropdown context menu to allow users to modify the animation
    void constructContextMenu(QPoint pos, int column);

    //! Get the animation framerange
    FrameRange getFrameRange() const { return itsMasterFrameRange; }

    //! Find the bounding keyframes for a given framenumber
    /*! If the requested frame is either before the first keyframe, or after the last,
        then both iterators in the pair will just be set to itsKeyframes.end()*/
    std::pair<QMap<int, FrameState>::const_iterator,
              QMap<int, FrameState>::const_iterator>
        getBoundingKeyframes(int fnum);

    AnimationDelegate::FrameType getFrameType(int fnum);

    QMap<int, FrameState>& getKeyFrames() { return itsKeyFrames; }

    //! Clear all keyframes from the keyframe storage
    void clear();
  signals:
    void animationChanged();

  private:
    QMap<int, FrameState> itsKeyFrames;

    FrameRange itsMasterFrameRange;
};

#endif //OBJECTANIMATION_QT_H

