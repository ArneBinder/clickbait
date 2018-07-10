# Clickbait

Train and use neural models for clickbait prediction, see [Clickbait Challenge](https://www.clickbait-challenge.org/).

## Usage

### Prerequisites and Installation

This how-to uses `git` and `miniconda`. If these are already installed, skip the following commands and continue
with `create python 3 environment`.

```bash
# install git
sudo apt update --yes && sudo apt install git --yes

# Get Miniconda and make it the main Python interpreter
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
rm ~/miniconda.sh
# add to PATH and reload .bash_profile to update PATH
echo "PATH=\$PATH:\$HOME/miniconda/bin" >> .bash_profile
source ~/.bash_profile
```


```bash
# create python 3 environment (named `clickbait`) with conda (and install pip into it)
conda create -n clickbait python=3 pip --yes

# get clickbait project
git clone https://github.com/ArneBinder/clickbait.git

# move into the project folder and install requirements into the new conda environment
cd clickbait
conda install -n clickbait --file requirements.txt --yes

# activate the new clickbait environment
source activate clickbait

# install the spacy model
pip install https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.0.0/en_vectors_web_lg-2.0.0.tar.gz#en_vectors_web_lg

```

## Prediction

This repo includes two trained models, [noimages](models/noimages) and [wimages](models/wimages). They are trained on
[Clickbait Challenge](https://www.clickbait-challenge.org/) data
([clickbait17-train-170630.zip](http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-train-170630.zip) and [clickbait16-train-170331.zip](http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait16-train-170331.zip)
was used for early stopping). The architecture consists of bi-GRUs followed by dense layers. [wimages](models/wimages)
additionally uses [InceptionResNetV2](https://keras.io/applications/#inceptionresnetv2) to embed image data,
if available, that is concatenated with the bi-GRU output.

The models can be used to get clickbait predictions by calling:
```bash
# MODEL_DIRECTORY: directory containing the files model_config.json and model_weights
# INPUT_DIRECTORY: directory containing data such as presented at [here](https://www.clickbait-challenge.org/#data) (the zip file has to be uncompressed)
# OUTPUT_DIRECTORY: the result is written here
python clickbait.py --model-dir <MODEL_DIRECTORY> -i <INPUT_DIRECTORY> -o <OUTPUT_DIRECTORY>
```

