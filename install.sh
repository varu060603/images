#!/bin/bash
git clone https://github.com/nngokhale/optimum-habana/
cd optimum-habana
git checkout GaudiQwen2VLShare123
cd ..
pip install ./optimum-habana
pip install -r ./optimum-habana/examples/language-modeling/requirements.txt
