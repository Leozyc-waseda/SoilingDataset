#ifndef ANIMATION_DELEGATE_QT_H
#define ANIMATION_DELEGATE_QT_H


#include <QtGui/QtGui>

class AnimationDelegate : public QAbstractItemDelegate
{
  Q_OBJECT

  public:
    AnimationDelegate(QObject *parent = 0);

    void paint(QPainter *painter, const QStyleOptionViewItem &option,
        const QModelIndex &index) const;

    QSize sizeHint(const QStyleOptionViewItem &option,
        const QModelIndex &index ) const;

    enum FrameType {Invisible, VisibleKeyframe, InvisibleKeyframe, Tween};

    void setSelectedRow(int rowIdx) {itsSelectedRow = rowIdx; }

  public slots:
    void setFrameType(FrameType type);
    void frameIndexChanged(int);

  private:
    FrameType itsFrameType;
    int itsCurrentFrameIndex;
    int itsSelectedRow;

};

#endif //ANIMATION_DELEGATE_QT_H
