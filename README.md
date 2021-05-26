# jatayu_open-unmix

- Clone this repo using 
 ```sh
 git clone https://github.com/MaulikGupta/jatayu_open-unmix
 ```
-
- Create a conda env using 
 ```sh
 environment-gpu-linux-cuda10.yml
 ``` 
  > ```conda env create -f scripts/environment-{INSERT CORRECT FILE}.yml```  
  > The various files which you can choose from are: 
  >  - cpu-linux
  >  -  gpu-linux-cuda10
  >  -  cpu-osx 
  > ___Choose the file depending on your system or else the model might not work.___
- Install Jupyter Notebook using 
```sh 
conda install -c conda-forge notebook
```
- 
- ```open-unmix-512``` is the model trained via open-unmix to separate crow calls; and ```open-unmix-titli``` is the model trained to separate sound of the bird blackbird.
    > __NOTE__ : To train your own model use command saved in ```audio_source_model_train_command.txt```.  Correct all file paths in jupyer norebooks run ```final.ipynb```

- In folder ```datasets/```> folders ```crow/```, ```sparrow/```, ```titli/```, ```ESC-50/``` are original datasets used for audio separation and classification
- To split original .mp3 files into 5 sec clips and to convert them into __wav__ format use ```sourcefolder_formatting.py```
- 
- For audio classification (DL) need to update ```meta.csv``` in ```audio classifcation/``` folder for new dataset location 
- For image classification :
      - Run ```crow_sparrow_img_classify.ipynb``` to train and save the model.
- For using saved model ```trained-model.ipynb```
  > __NOTE :__ In file ```data.py``` at ___line 406___ you must change = to >= (or while training you can tell set the input lenght lesser than the actual length of the audio files)
