
PreferencesDialog::PreferencesDialog(QWidget *parent) :
  QDialog(parent)
{

  QFormLayout *layout = new QFormLayout;

  archiveLocEdit = new QLineEdit("/lab/dbvideos/");
  layout->addRow(tr("&Archive Location"), archiveLocEdit);

  workingLocEdit = new QLineEdit("/lab/dbvideos/");
  layout->addRow(tr("&Working Location"), workingLocEdit);

  incomingLocEdit = new QLineEdit("/lab/dbvideos/incoming/");
  layout->addRow(tr("&Incoming Location"), incomingLocEdit);

  QPushButton* cancelButton = new QPushButton("&Cancel", this);
  connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
  layout->addRow(cancelButton);

  QPushButton* connectButton = new QPushButton("&Save", this);
  connect(connectButton, SIGNAL(clicked()), this, SLOT(accept()));
  layout->addRow(connectButton);

  setLayout(layout);
}
