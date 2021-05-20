# jatayu_open-unmix

1. clone this repo
2. clone open-unmix template in it
3. create conda env using 'environment-gpu-linux-cuda10.yml'  --> conda env create -f scripts/environment-X.yml where X is either [cpu-linux, gpu-linux-cuda10, cpu-osx], depending on your system
4. from 'open-unmix crack' folder copy all files to 'open-unimx-pytorch' folder i.e. the cloned template for open-unmix
5. 'open-unmix-512' is the model trained via open-unmix to separate crow calls; and 'open-unmix-titli' is the model trained to separate sound of the bird blackbird.
    to train your own model use command saved in 'audio_source_model_train_command.txt'. 


6. in folder 'datasets', folders 'crow', 'sparrow', 'titli', 'ESC-50' are original datasets used for audio separation and classification
7. to split original mp3 files into 5 sec clips and to convert them into wav format use 'sourcefolder_formatting.py'
8. 
9. for audio classification (DL) need to update meta.csv in 'audio classifcation' folder for new dataset location 
10. for image classification:-
      run 'crow_sparrow_img_classify.ipynb' to train and save the model.
      for using saved model 'trained-model.ipynb'
10. data.py --> line 406 >=
