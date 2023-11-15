#!/bin/sh

# run this AS A NORMAL USER to configure your home for beobot use:

# add some short cut into .bashrc
read -p "Do you want tune .bashrc and .vimrc for Beobot(Y/n)?"
if [ "$REPLY" == "y" ]; then
    echo "export EDITOR=vim" >> ~/.bashrc
    echo "alias cdr='cd ~/saliency/src/Robots/Beobot2/Hardware/;pwd'" >> ~/.bashrc
fi

#add vim setting
if [ "$REPLY" == "y" ]; then
    echo "set ts=2" >> ~/.vimrc
    echo "set sw=2" >> ~/.vimrc
    echo "filetype on" >> ~/.vimrc
    echo "au BufNewFile,BufRead *.spin set filetype=spin" >> ~/.vimrc
    #sudo cp ./vim/spin.vim /usr/share/vim/syntax/
fi

