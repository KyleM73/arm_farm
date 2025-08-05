#!/bin/bash
# create a symbolic link to the system's Python library, which is
# required by MuJoCo's passive viewer on macOS.

# Get the Python version like "3.11"
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
DYLIB_NAME="libpython${PYTHON_VERSION}.dylib"

DYLIB_DEST="$VIRTUAL_ENV/lib/$DYLIB_NAME"

# only create once
if [ ! -L "$DYLIB_DEST" ]; then
  LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
  DYLIB_SOURCE="$LIBDIR/$DYLIB_NAME"
  
  if [ -f "$DYLIB_SOURCE" ]; then
    echo "Linking $DYLIB_SOURCE to $DYLIB_DEST"
    ln -s "$DYLIB_SOURCE" "$DYLIB_DEST"
  else
    echo "Warning: Python library not found at '$DYLIB_SOURCE'. MuJoCo viewer may not work."
  fi
fi