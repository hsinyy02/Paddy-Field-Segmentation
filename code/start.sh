# chmod +x start.sh
#!/bin/bash

python get_paddy.py
python detect_seedling.py
python label_seedling.py
python stitching.py
python align_seedling.py