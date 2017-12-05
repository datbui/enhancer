#!/usr/bin/env bash

echo 'Converting images to tfrecord files'
python tfrecords.py --tfrecord_mode=create
echo 'Tfrecords have been created'