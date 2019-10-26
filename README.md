# Modeling Personality with Attentive Networks and Contextual Embeddings

## Introduction
This repository is for a paper accepted for AAAI 2020 student abstract. In this project, we introduced both attentive networks and contextual embeddings (BERT and roBERTa) to the task and explored the performance on two datasets: "FriendsPersona" and "Essays". 

We created the first dialogue dataset "FriendsPersona" for automatic personality recognition with a novel and scalable dialogue extraction algorithm. We established a solid benchmark on our "FriendsPersona" dataset. And we achieved state-of-art results on the benchmark "Essays" dataset. 

Automatic text-based personality recognition, as an important topic in computational psycho-linguistics, focuses on determining one's personality traits from text. The Big Five Hypothesis is usually used for measuring one's personality in five binary traits: agreeableness (AGR), conscientiousness (CON), extraversion (EXT), openness (OPN), neuroticism (NEU).

## Dataset
### Essays
We achieved state-of-art results on this dataset. [Essays](https://psycnet.apa.org/doiLanding?doi=10.1037%2F0022-3514.77.6.1296) dataset is the benchmark dataset for text-based personality recognition with 2,468 self-report essays.

### Friends Personality
We created this dataset and established a solid benchmark. 711 short conversations are extracted and annotated from the first four seasons of Friends TV Show transcripts.

You could find more details about this dataset in this repo [emorynlp/personality-detection](https://github.com/emorynlp/personality-detection).

## Models and Experiments
We experimented on both dataset with ABCNN, ABLSTM, HAN, BERT, RoBERTa models.

You could find models and training process about ABCNN, ABLSTM, HAN in `/keras_models/main.py`.

We use huggingface's [pytorch-transformer](https://github.com/huggingface/pytorch-transformers) to fine tune BERT and RoBERTa. You could find more details in `/huggingface/pytorch-transformers-master/examples/run_glue.py`


### Results on Essays
![](https://drive.google.com/uc?export=view&id=1Mio-FHNcMYILayHpmKTJSdFHRsyRjd2T)


### Results on Friends Personality
![](https://drive.google.com/uc?export=view&id=1pVZC-ga2dt_PecUEfwMkCFRWaLFJMbla)
