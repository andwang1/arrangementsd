#!/bin/bash
./waf configure --exp balltrajectorysd --cpp14=yes --kdtree /workspace/include

./waf --exp balltrajectorysd -j 1
echo 'FINISHED BUILDING. Now fixing name of files'
python -m fix_build --path-folder /git/sferes2/build/exp/balltrajectorysd/
