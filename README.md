# jatayu_open-unmix

1. clone this repo
2. clone open-unmix template in it
3. in folder 'datasets', folders 'crow', 'sparrow', 'titli', 'ESC-50' are original datasets used
4. to split original mp3 files into 5 sec clips and to convert them into wav format use 'sourcefolder_formatting.py'
5. for audio classification (DL) need to update meta.csv in 'audio classifcation' folder for new dataset location 
6. create conda env using 'environment-gpu-linux-cuda10.yml'
7. data.py --> line 406 >=
