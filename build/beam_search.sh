cd ~/HandwritingDetection/build/app/models/OCRBeamSearch/src/

git clone https://github.com/githubharald/CTCWordBeamSearch.git
cd CTCWordBeamSearch/cpp/proj/

NUMTHREADS="4"
echo "Parallel decoding with $NUMTHREADS threads"
PARALLEL="-DWBS_PARALLEL -DWBS_THREADS=$NUMTHREADS"

# get and print TF version
TF_VERSION=$(python3 -c "import tensorflow as tf ;  print(tf.__version__)")
echo "Your TF version is $TF_VERSION"
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# compile
g++ -Wall -O2 --std=c++11 -shared -o TFWordBeamSearch.so ../src/TFWordBeamSearch.cpp ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} $PARALLEL

# copy TFWordBeamSearch.so
cp CTCWordBeamSearch/cpp/proj/TFWordBeamSearch.so .


cd ~/HandwritingDetection/build/app/models/OCRBeamSearch/model/
unzip model.zip
# rm model.zip