#!/bin/bash

URL="http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip"

echo "Downloading $URL..."
curl -L "$URL"

if [ $? -ne 0 ]; then
    echo "Download failed."
    exit 1
fi

echo "Unzipping..."
unzip "Multivariate2018_ts.zip"

echo "Done."
