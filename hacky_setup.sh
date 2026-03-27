__doc__="
This is what I had to do get to get this repo working for shitspotter in its
forked state.

This just builds the deformable attention module, it doesn't set this repo up
in dev mode. I don't think that's possible right now.
"
uv pip install -r tmp_requirements.txt

cd models/GroundingDINO/ops
# Modified 2 places in ~/code/Open-GroundingDino/models/GroundingDINO/ops/src/cuda/ms_deform_attn_cuda.cu
# to go from type() -> scalar_type() to make this work on 3.13 with torch 2.7.1+cu126, nvcc cuda_12.0.r12.0/compiler.32267302_0
python setup.py build_ext --inplace -v
# IDK why, this can't find required libs. Force it to.
TORCH_LIB_DPATH=$(dirname $(find $(python -c "import torch; print(torch.__path__[0])") -name "libc10.so"))
export LD_LIBRARY_PATH=$TORCH_LIB_DPATH:$LD_LIBRARY_PATH
ldd MultiScaleDeformableAttention*
# Test we can import
python -c "import MultiScaleDeformableAttention"
cd ../../..
cp models/GroundingDINO/ops/MultiScaleDeformableAttention.*.so .
python -c "import MultiScaleDeformableAttention"

