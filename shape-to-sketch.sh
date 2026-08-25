python shape-to-sketch.py -i "$1" -o "${1%.*}.latk" --min-segment-points 4 --n-surface 2048 --curvature-threshold 1.0 --salient-mode "both" --n-salient 6000 --thin-iterations 1    
