import os 
import urllib.request 
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

fileroot = '/mnt/home/20170419/ssl/meta/' #set your location
metafile = 'kaist_naver_prod200k_class265_meta_alldata.txt'

temp_folder = os.path.join(fileroot,'temp')
savefolder = os.path.join(fileroot,'images')
nFilecnt = 0

if not os.path.exists(temp_folder): os.makedirs(temp_folder)
if not os.path.exists(savefolder): os.makedirs(savefolder)
                
with open(os.path.join(fileroot,metafile), 'r') as rf:
    for idx, line in (enumerate(tqdm(rf))):
        if idx ==0:
            continue  # skip firts line
        
        # step1. load meta inforation
        instance = line.split('\t')[0]
        img_urls = line.split('\t')[1]
        save_filename =  line.split('\t')[2].split('\n')[0]
        
        # step2. download image to temporal folder 
        tempfilepath = os.path.join(temp_folder, save_filename )
        urllib.request.urlretrieve(img_urls, tempfilepath)
        
        # step3. load temporal down loaded image 
        img = Image.open(tempfilepath)    
        img = img.convert("RGB")
        
        # step4. save images
        finalfilepath = os.path.join(savefolder, save_filename )
        img.save(finalfilepath, 'jpeg') 
        nFilecnt += 1    
        
        # step5. remove temporally saved image
        os.remove(tempfilepath)           
        
print(nFilecnt )            