gzip -d train-images-idx3-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
dd iflag=skip_bytes skip=16 if=train-images-idx3-ubyte > images1
dd iflag=skip_bytes skip=16 if=t10k-images-idx3-ubyte > images2
dd iflag=skip_bytes skip=8 if=train-labels-idx1-ubyte > labels1
dd iflag=skip_bytes skip=8 if=t10k-labels-idx1-ubyte > labels2
cat images1 images2 > images
cat labels1 labels2 > labels
rm images1 images2 labels1 labels2





