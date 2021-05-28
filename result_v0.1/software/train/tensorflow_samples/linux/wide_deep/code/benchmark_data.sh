echo "Begin DownLoad Criteo Data"
wget --no-check-certificate https://paddlerec.bj.bcebos.com/benchmark/criteo_benchmark_data.tar.gz
echo "Begin Unzip Criteo Data"
tar -xf criteo_benchmark_data.tar.gz
echo "Only keep origin test data"
rm -rf train_data

echo "Begin DownLoad TF Criteo Data"
wget --no-check-certificate https://paddlerec.bj.bcebos.com/benchmark/tf_criteo.tar.gz
echo "Begin Unzip TF Criteo Data"
tar -xf tf_criteo.tar.gz

mv tf_criteo/train_data/ ./
echo "Get TF Criteo Data Success"
