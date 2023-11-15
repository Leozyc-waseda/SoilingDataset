#ifndef ANIMATION_DELEGATE_QT_C
#define ANIMATION_DELEGATE_QT_C


#include "NeovisionII/NeoAnnotate/AnimationDelegate.qt.H"
#include "Util/log.H"


AnimationDelegate::AnimationDelegate(QObject *parent)
{
  itsFrameType = AnimationDelegate::Invisible;
}

void AnimationDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option,
        const QModelIndex &index) const
{

  QRect rect = option.rect;

  QColor         currFrameColor(200, 255, 255);
  QColor          selectedColor(225, 225, 255);
  QColor selectedCurrFrameColor(195, 225, 225);

  if(option.state & QStyle::State_Selected)
  {
    painter->fillRect(rect, option.palette.highlight());
  }
  else
  {
    if(index.row() == itsSelectedRow)
    {
      if(itsCurrentFrameIndex == index.column())
        painter->fillRect(rect, selectedCurrFrameColor);
      else
        painter->fillRect(rect, selectedColor);
    }
    else if(itsCurrentFrameIndex == index.column())
      painter->fillRect(rect, currFrameColor);
  }


  FrameType type = FrameType(index.model()->data(index, Qt::DisplayRole).toInt());
  if(type == Invisible)
    return;


  painter->setRenderHint(QPainter::Antialiasing, true);
  //painter->setBrush(QBrush(Qt::black));
  if(type == VisibleKeyframe)
  {
    int radius = 4;
    painter->setBrush(QColor(0, 255, 0));
    painter->drawEllipse(QRectF(rect.x() + rect.width()/2 - radius,
                                 rect.y() + rect.height()/2 - radius,
                                 2*radius, 2*radius));
  }
  if(type == InvisibleKeyframe)
  {
    int radius = 4;
    painter->setBrush(QColor(255, 0, 0));
    painter->drawEllipse(QRectF(rect.x() + rect.width()/2 - radius,
                                 rect.y() + rect.height()/2 - radius,
                                 2*radius, 2*radius));
  }
  else if(type == Tween)
  {
    painter->drawLine(rect.x(),  rect.y() + rect.height()/2,
                      rect.x() + rect.width(), rect.y() + rect.height()/2);
  }
}

QSize AnimationDelegate::sizeHint(const QStyleOptionViewItem & /* option */,
    const QModelIndex & /* index */) const
{
  return QSize(10,10);
}

void AnimationDelegate::setFrameType(FrameType type)
{
  itsFrameType = type;
}

void AnimationDelegate::frameIndexChanged(int fnum)
{
  itsCurrentFrameIndex = fnum;
}

#endif //ANIMATION_DELEGATE_QT_C
