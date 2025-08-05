#!/bin/bash
ACTIVATE_SCRIPT_PATH=$1
POSTACTIVATE_LOGIC='# Run postactivate scripts if present
if [ -d "$VIRTUAL_ENV/bin/postactivate.d" ]; then
  for f in "$VIRTUAL_ENV"/bin/postactivate.d/*; do
    # Source the script if it is executable
    [ -x "$f" ] && . "$f"
  done
fi'

# only add once
if ! grep -q 'postactivate.d' "$ACTIVATE_SCRIPT_PATH"; then
  echo "Patching activate script to run post-activation scripts."
  printf "\n%s\n" "$POSTACTIVATE_LOGIC" >> "$ACTIVATE_SCRIPT_PATH"
fi