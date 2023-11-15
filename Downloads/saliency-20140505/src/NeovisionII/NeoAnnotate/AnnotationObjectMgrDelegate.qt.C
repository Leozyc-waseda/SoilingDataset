#include "NeovisionII/NeoAnnotate/AnnotationObjectMgrDelegate.qt.H"

AnnotationObjectMgrDelegate::AnnotationObjectMgrDelegate(QObject *parent)
{

}

void AnnotationObjectMgrDelegate::paint(QPainter* painter,
    const QStyleOptionViewItem & option, const QModelIndex & index) const
{
  QRect rect = option.rect;
  QString text;

  switch(index.column())
  {
    case 0:
    case 1:
      text = index.data().toString();
      break;
    case 2:
      text = itsCategories[index.data().toInt()];
      break;
  }

  painter->drawText(rect, Qt::AlignCenter, text);
}

QSize AnnotationObjectMgrDelegate::sizeHint(const QStyleOptionViewItem & option, 
    const QModelIndex & index) const
{
  return QSize();
}

QWidget* AnnotationObjectMgrDelegate::createEditor(QWidget * parent, const QStyleOptionViewItem & option, const QModelIndex & index ) const
{
  QRect rect = option.rect;

  QLineEdit* lineEdit;
  QComboBox* comboBox;

  switch(index.column())
  {
    case 0:
      return NULL;
    case 1:
      lineEdit = new QLineEdit(parent);
      lineEdit->setGeometry(rect);
      return lineEdit;
    case 2:
      comboBox = new QComboBox(parent);
      comboBox->setGeometry(rect);
      QMap<int, QString>::const_iterator catIt = itsCategories.constBegin();
      for(;catIt != itsCategories.constEnd(); ++catIt)
        comboBox->addItem(catIt.value(), catIt.key());
      return comboBox;
  }
  return 0;
}

void AnnotationObjectMgrDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
}

void AnnotationObjectMgrDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
  QLineEdit* lineEdit;
  QComboBox* comboBox;
  QString data;

  switch(index.column())
  {
    case 1:
      lineEdit = static_cast<QLineEdit*>(editor);
      data = lineEdit->text();

      break;
    case 2:
      comboBox = static_cast<QComboBox*>(editor);
      data = comboBox->itemData(comboBox->currentIndex()).toString();
      break;
  }
  qDebug() << "Setting Model Data: " << data;
  model->setData(index, data, Qt::EditRole);
}
