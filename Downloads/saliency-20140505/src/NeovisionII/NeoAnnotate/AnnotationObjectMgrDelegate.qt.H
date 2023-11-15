#ifndef NEOVISIONII_NEOANNOTATE_ANNOTATIONOBJECTMGRDELEGATE_QT_H
#define NEOVISIONII_NEOANNOTATE_ANNOTATIONOBJECTMGRDELEGATE_QT_H

#include <QtGui/QtGui>

class AnnotationObjectMgrDelegate : public QAbstractItemDelegate
{
  Q_OBJECT

  public:
    AnnotationObjectMgrDelegate(QObject *parent = 0);

    void paint(QPainter* painter,
        const QStyleOptionViewItem & option, const QModelIndex & index) const;

    QSize sizeHint(const QStyleOptionViewItem & option, 
        const QModelIndex & index) const;

    QWidget* createEditor(QWidget * parent, const QStyleOptionViewItem & option, const QModelIndex & index ) const;

    void setEditorData(QWidget *editor, const QModelIndex &index) const;

    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;

    void setObjCategories(QMap<int, QString> categories) { itsCategories = categories; }

  private:
    QMap<int, QString> itsCategories;


};

#endif // NEOVISIONII_NEOANNOTATE_ANNOTATIONOBJECTMGRDELEGATE_QT_H

