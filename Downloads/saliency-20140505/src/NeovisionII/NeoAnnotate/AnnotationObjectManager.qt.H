#ifndef ANNOTATIONOBJECTMANAGER_H
#define ANNOTATIONOBJECTMANAGER_H

#include <QtCore/qabstractitemmodel.h>

#include "NeovisionII/NeoAnnotate/AnnotationObject.qt.H"

class AnnotationObjectManager : public QAbstractTableModel
{
  Q_OBJECT

  public:
    //! Construct a new Annotation Object Manager inside of parent
    AnnotationObjectManager(QObject * parent = 0);

    //! Get the number Annotation Objects being managed by this manager
    int rowCount(const QModelIndex &parent = QModelIndex()) const;

    //! Get the number of data items to report for each Annotation Object
    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    //! Get a data item from the object list
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;

    //! Get the header names
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;

    //! Set a particular data member of one of our Annotation Objects
    bool setData(const QModelIndex &index, const QVariant &value, int role);

    //! Return the flags of a particular row/column cell
    Qt::ItemFlags flags(const QModelIndex &index) const;

    //! Add an object to our object list
    bool addObject(AnnotationObject * object);

    //! Get an object reference by it's ID, or NULL if no object exists with that ID
    AnnotationObject* getObjectById(int id);

    //! Sets itsAnimationView
    void setAnimationView(QTableView *animationView);

    //! Pass a request to construct a menu off to the current selected object
    void constructAnimationContextMenu(QPoint globalPos, int row, int column);

    //! Clear the entire storage, deleting all existing Annotation Objects
    void clear();

    void setAnimationRowSelected(int rowIdx);

    void setLastAnimViewClick(const QModelIndex & pos);

    std::map<int, std::map<int, AnnotationObjectFrame> > renderAnimations();

    QList<AnnotationObject *> getAnnotationObjects() { return itsObjects; }

  signals:
    //! Emit a signal to let the view know that a particular object has been selected
    /*! The integer parameter indicates the item's row. */
    void selectingObject(int);

  public slots:
    //! select() is called when an item is selected in the table
    /*! In this method we ensure that our graphicsview and tableview are in sync
        and that only one item is selected at a time */
    void select(const QModelIndex & index);

    //! Inform the manager that an item is being selected in the graphicsview
    /*! This highlights and selects the object*/
    void objectSelected(int itemId);

    //! Add a vertex to the selected object
    void addVertex(QPointF);

    //! Remove a vertex from the selected object
    void removeVertex(QPointF);

    //! Increase the size of the selected object
    void expandPolygon();

    //! Reduce the size of the selected object
    void shrinkPolygon();

    //! Stretch the current polygon by this amount
    void stretchPolygon(QPointF);

    //! Make the next frame invisible for the current object
    void endTrack();

    //! Remove an object
    void removeObject();

    //! The frame number has changed
    void frameChanged(int fnum);

    //! Select a row in the animation view
    void selectAnimationRow(int rowIdx);

    //! Set the opacity of all objects
    void setOpacity(int opacity);

  private:
    //! Select the given object, and deselect the last selected object
    void selectObject(int rowIdx);

    //! Deselect the given object
    void deselectObject(int rowIdx);

    int itsCurrentFrame;                  //!< The current frame number

    QList<AnnotationObject *> itsObjects; //!< Storage for this manager's Annotation Objects

    int itsCurrentSelectedRow;            //!< The currently selected row,
                                          //!< or -1 if no row is selected

    int itsOpacity;                       //!< The current user selected opacity

    QTableView * itsAnimationView;        //!< A pointer to the QTableView which displays
                                          //!< the current animation in the MainWindow
};

#endif //ANNOTATIONOBJECTMANAGER_H


