#!/bin/bash -x

# install the dejavu fonts
# sudo pacman -S --noconfirm ttf-dejavu

# symlink the ttfs to the location where qt expects them
mkdir -p "$VIRTUAL_ENV/lib/python3.12/site-packages/cv2/qt/fonts"
ln -s /usr/share/fonts/TTF/DejaVu*.ttf "$VIRTUAL_ENV/lib/python3.12/site-packages/cv2/qt/fonts/" 2>/dev/null