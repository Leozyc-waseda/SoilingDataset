#ifndef ANIMATIONMODEL_QT_H
#define ANIMATIONMODEL_QT_H

#include <QtGui/QtGui>
#include "Media/FrameRange.H"

class AnnotationObject;
class AnnotationObjectVertex;

class AnimationModel : public QAbstractTableModel
{
  Q_OBJECT

  public:
    AnimationModel(AnnotationObject *parent);

    //! Get the header names
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;

    //! Let our view know how many rows to display
    int rowCount(const QModelIndex &parent = QModelIndex()) const;

    //! Get the number framers in the animation
    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    //! Get a data item from the object list
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;

    //! Set a particular frame of this animation
    bool setData(const QModelIndex &index, const QVariant &value, int role);

    //! Return the flags of a particular row/column cell
    Qt::ItemFlags flags(const QModelIndex &index) const;

    //! Let Qt know that we support move operations
    Qt::DropActions supportedDropActions() const;

    //! Insert a new keyframe at frameNumber indicating the object is at position pos
    void setPosition(int frameNum, QPointF pos);

    //! Called when a keyframe is dragged to a new position
    bool dropMimeData(const QMimeData * data,
        Qt::DropAction action,
        int row, int col,
        const QModelIndex & parent);

    //! A wrapper around the protected QAbstractTableModel::beginInsertRows function
    void beginInsertRow(int row);

    //! A wrapper around the protected QAbstractTableModel::endInsertRows function
    void endInsertRow();

    //! A wrapper around the protected QAbstractTableModel::beginRemoveRows function
    void beginRemoveRow(int row);

    //! A wrapper around the protected QAbstractTableModel::endRemoveRows function
    void endRemoveRow();

    //! A wrapper around the protected QAbstractTableModel::beginRemoveRows function
    void beginRemoveRowsPub(const QModelIndex & i, int first, int last);

    void setLastClick(const QModelIndex & index) { itsLastClick = index; }

  public slots:
    //! Whenever any aspect of the animation has changed, this slot gets called to redraw
    //! the whole animation view. This should be reimplemented later to only redraw the
    //! changed sections.
    void animationChanged();

  private:
    AnnotationObject * itsParent;                //!< The AnnotationObject which this model represents

    QList<AnnotationObjectVertex*> *itsVertices; //!< Convenience pointer to the objects vertices

    QPoint itsLastMousePress;

    QModelIndex itsLastClick;

};

#endif //ANIMATIONMODEL_QT_H

