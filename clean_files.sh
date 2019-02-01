find . -name "*.so" | xargs rm -rf
find . -name "*.pyc" | xargs rm -rf
find . -name "build" | xargs rm -rf
find . -name "__pycache__" | xargs rm -rf