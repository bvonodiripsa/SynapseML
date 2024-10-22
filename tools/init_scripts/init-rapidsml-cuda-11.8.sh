# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
# IMPORTANT: specify RAPIDS_VERSION fully 23.10.0 and not 23.10
# also in general, RAPIDS_VERSION (python) fields should omit any leading 0 in month/minor field (i.e. 23.8.0 and not 23.08.0)
# while SPARK_RAPIDS_VERSION (jar) should have leading 0 in month/minor (e.g. 23.08.2 and not 23.8.2)
RAPIDS_VERSION=24.6.0
# SPARK_RAPIDS_VERSION=23.10.0
SPARK_RAPIDS_VERSION=24.08.1
SPARK_RAPIDSML_VERSION=24.6.0
TORCH_VERSION=2.4.0
TRT_VERSION=10.4.0
MODEL_NAVIGATOR_VERSION=0.10.1
SENTENCE_TRANSFORMER_VERSION=3.2.0
URLLIB3_VERSION=1.25.11
CYTHON_VERSION=3.0.11
MPI4PY_VERSION=4.0.1
TRT_LLM=0.13.0
TRANSFORMER_VERSION=4.42.4
PYMUPDF=1.24.11

curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${SPARK_RAPIDS_VERSION}/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}-cuda11.jar -o /databricks/jars/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar

# install cudatoolkit 11.8 via runfile approach
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# reset symlink and update library loading paths
rm /usr/local/cuda
ln -s /usr/local/cuda-11.8 /usr/local/cuda

# upgrade pip
/databricks/python/bin/pip install --upgrade pip

# install cudf, cuml and their rapids dependencies
# using ~= pulls in latest micro version patches
/databricks/python/bin/pip install cudf-cu11~=${RAPIDS_VERSION} \
    cuml-cu11~=${RAPIDS_VERSION} \
    pylibraft-cu11~=${RAPIDS_VERSION} \
    rmm-cu11~=${RAPIDS_VERSION} \
    --extra-index-url=https://pypi.nvidia.com

# onnx runtime for CU11
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

# torch for CU11
pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu118


# install model navigator
/databricks/python/bin/pip install --extra-index-url https://pypi.nvidia.com onnxruntime-gpu tensorrt==${TRT_VERSION} triton-model-navigator==${MODEL_NAVIGATOR_VERSION} sentence_transformers urllib3==${URLLIB3_VERSION} 

# remove CU12
pip uninstall tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs nvidia_cuda_runtime_cu12  -y

# install spark-rapids-ml
/databricks/python/bin/pip install spark-rapids-ml~=${SPARK_RAPIDSML_VERSION}

# install TRT-LLM
/databricks/python/bin/pip install --upgrade cython==${CYTHON_VERSION}
/databricks/python/bin/pip install --pre --no-build-isolation --extra-index-url https://pypi.nvidia.com mpi4py==${MPI4PY_VERSION}
# /databricks/python/bin/pip install --pre --extra-index-url https://pypi.nvidia.com tensorrt-llm==0.12.0.dev2024073000
/databricks/python/bin/pip install --pre --extra-index-url https://pypi.nvidia.com tensorrt-llm==${TRT_LLM}

# Required by NY-Embed
/databricks/python/bin/pip install --upgrade sentence-transformers==${SENTENCE_TRANSFORMER_VERSION}
/databricks/python/bin/pip install transformers==${TRANSFORMER_VERSION}

# To work with PDF
/databricks/python/bin/pip install PyMuPDF==${PYMUPDF}

cp -f /dbfs/FileStore/files/executor.py /databricks/python3/lib/python3.10/site-packages/tensorrt_llm
