#!/bin/bash

if [ "$(uname | cut -f 1 -d_)" == Linux ]; then
	# We need libGL.so.1 installed on Linux. The given approach will work on Debian-based systems,
	# such as Ubuntu. On non-Debian systems, it will fail, we will print the warning, and then hope
	# that the user already has it installed (which is likely, but not guaranteed, especially on a
	# headless setup).
	apt-get install -y libgl1 || echo "Unable to install libGL.so.1. We hope it's already there..."
fi
