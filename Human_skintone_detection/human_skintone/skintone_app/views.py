from django.shortcuts import render
from .forms import ImageForm
from .models import Image

def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        #title1 = form.title

        if form.is_valid():
            
            form.save()

            id1 = Image.objects.latest("id")
            path1 = (id1.image.path)
            print("path is ",path1)
            print(type(id1))

             



           #ML code starts here

           
            # Extracting RGB Value of Skin

            import numpy as np
            import cv2
            from sklearn.cluster import KMeans
            from collections import Counter
            import imutils
            from matplotlib import pyplot as plt


            rgb_lower = [141,85,36]
            rgb_higher = [255,219,172]

            skin_shades = {
                'dark' : [rgb_lower,[198,134,66]],
                'mild' : [[198,134,66],[224,172,105]],
                'fair':[[224,172,105],[241,194,125]],
                'bright' : [[241,194,125],rgb_higher]
            }

            convert_skintones = {}
            for shade in skin_shades:
                convert_skintones.update({
                    shade : [
                        (skin_shades[shade][0][0] * 256 * 256) + (skin_shades[shade][0][1] * 256) + skin_shades[shade][0][2],
                        (skin_shades[shade][1][0] * 256 * 256) + (skin_shades[shade][1][1] * 256) + skin_shades[shade][1][2]
                    ]
                })

            def extractSkin(image):
                img = image.copy()
                black_img = np.zeros((img.shape[0],img.shape[1],img.shape[2]),dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
                print(lower_threshold)
                upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
                print(upper_threshold)

                skinMask = cv2.inRange(img, lower_threshold, upper_threshold)
                skin = cv2.bitwise_and(img, img, mask=skinMask)
                return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

            def removeBlack(estimator_labels, estimator_cluster):
                hasBlack = False
                occurance_counter = Counter(estimator_labels)
                def compare(x, y): return Counter(x) == Counter(y)
                for x in occurance_counter.most_common(len(estimator_cluster)):
                    color = [int(i) for i in estimator_cluster[x[0]].tolist()]
                    if compare(color, [0, 0, 0]) == True:
                        del occurance_counter[x[0]]
                        hasBlack = True
                        estimator_cluster = np.delete(estimator_cluster, x[0], 0)
                        break
                return (occurance_counter, estimator_cluster, hasBlack)

            def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):
                occurance_counter = None
                colorInformation = []
                hasBlack = False
                if hasThresholding == True:
                    (occurance, cluster, black) = removeBlack(
                        estimator_labels, estimator_cluster)
                    occurance_counter = occurance
                    estimator_cluster = cluster
                    hasBlack = black
                else:
                    occurance_counter = Counter(estimator_labels)
                totalOccurance = sum(occurance_counter.values())
                for x in occurance_counter.most_common(len(estimator_cluster)):
                    index = (int(x[0]))
                    index = (index-1) if ((hasThresholding & hasBlack)
                                        & (int(index) != 0)) else index
                    color = estimator_cluster[index].tolist()
                    color_percentage = (x[1]/totalOccurance)
                    colorInfo = {"cluster_index": index, "color": color,
                                "color_percentage": color_percentage}
                    colorInformation.append(colorInfo)
                return colorInformation

            def extractDominantColor(image, number_of_colors=1, hasThresholding=False):
                if hasThresholding == True:
                    number_of_colors += 1
                img = image.copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.reshape((img.shape[0]*img.shape[1]), 3)
                estimator = KMeans(n_clusters=number_of_colors, random_state=0)
                estimator.fit(img)
                colorInformation = getColorInformation(
                    estimator.labels_, estimator.cluster_centers_, hasThresholding)
                return colorInformation

            def plotColorBar(colorInformation):
                color_bar = np.zeros((100, 500, 3), dtype="uint8")
                top_x = 0
                for x in colorInformation:
                    bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])
                    color = tuple(map(int, (x['color'])))
                    cv2.rectangle(color_bar, (int(top_x), 0),
                                (int(bottom_x), color_bar.shape[0]), color, -1)
                    top_x = bottom_x
                return color_bar

            def extractface(img):
                try:
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                    faces=detect.detectMultiScale(
                            gray,
                            scaleFactor=1.3,
                            minNeighbors=5
                    )
                    for (x,y,w,h) in faces:
                        face=img[y:y+h,x:x+w]
                        Cropped = cv2.resize(face, (400, 300))
                        return Cropped
                except:
                    return img

            #url = input("Enter image url :")

            #path=input("Enter path for the image :")
            image = cv2.imread(path1)

            image = imutils.resize(image, width=250)
            x=extractface(image)
            plt.subplot(3, 1, 1)

            try:
                plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
                image=x

            except:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
            plt.title("Original Image")

            skin = extractSkin(image)
            plt.subplot(3, 1, 2)
            plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
            plt.title("Thresholded  Image")

            unprocessed_dominant = extractDominantColor(skin, number_of_colors=1, hasThresholding=True)

            decimal_lower = (rgb_lower[0] * 256 * 256) + (rgb_lower[1] * 256) + rgb_lower[2]
            decimal_higher = (rgb_higher[0] * 256 * 256) + (rgb_higher[1] * 256) + rgb_higher[2]
            dominantColors = []
            for clr in unprocessed_dominant:
                clr_decimal = int((clr['color'][0] * 256 * 256) + (clr['color'][1] * 256) + clr['color'][2])
                if clr_decimal in range(decimal_lower,decimal_higher+1):
                    clr['decimal_color'] = clr_decimal
                    dominantColors.append(clr)

            print("Color Bar")
            colour_bar = plotColorBar(dominantColors)
            #plt.subplot(3, 1, 3)
            #plt.axis("off")

            #plt.imshow(colour_bar)
            plt.title("Color Bar")

            plt.tight_layout()
            #plt.show()


            rgb_value=colour_bar[0][0]

            rgb=list(rgb_value)
            rgb

            # Importing Dataset

            import pandas as pd
            import numpy as np

            list_g=[140,188,163,102,51,47,120,194,185,185,161,145,132,122,57,4,5,0,224,207,171,177,166,114,25,0,0,228,214,189,194,185,146,151,109,10,173,136,0,0,0,40,92,81,150,243,220,194,154,111,187,165,218,187,160,114,202,185,171,128,57,137,102,70,29,87,134,224,181,161,212,185,162,124,136,129,34,46,57,69,80,103,114,126,138,149,161,172,184,195,206,218,229,219,85,172,220,112,96,164,154,131,117,110,162,128,113,184,154,103,103,109,141,126,131,147,144,151,128,137,113,98,113,120,91,122,139,154,2,185,234,231,229,226,178,136,170,139,152,112,138,119,137,80,127,164,114,139,148,126,134,154,133,131,118,159,128,81,153,224,221,50,21,18,77,119,110,28,22,17,34,193,173,165,193,148,138,67,45,109,124,207,25,225,132,33,0,129,230,7,136,84,16,192,147,128,15,13,211,204,141,30,127,217,17,228,8,139,238,210,12,224,128,12,21,104,111,207,0,126,130,207,8,203,150,115,20,9,219,212,151,22,157,222,12,218,5,169,223,223,12,197,108,12,45,118,117,196,32,125,116,211,23,205,138,133,26,11,203,235,131,20,152,204,13,213,0,153,221,220,197,117,26,51,71,81,213,9,140,188,163,102,51,47,120,194,185,185,161,145,132,122,57,4,5,0,224,207,171,177,166,114,25,0,0,228,214,189,194,185,146,151,109,10,173,136,0,0,0,40,92,81,150,243,220,194,154,111,187,165,218,187,160,114,202,185,171,128,57,137,102,70,29,87,134,224,181,161,212,185,162,124,136,129,34,46,57,69,80,103,114,126,138,149,161,172,184,195,206,218,229,219,85,172,220,112,96,164,154,131,117,110,162,128,113,184,154,103,103,109,141,126,131,147,144,151,128,137,113,98,113,120,91,122,139,154,2,185,234,231,229,226,178,136,170,139,152,112,138,119,137,80,127,164,114,139,148,126,134,154,133,131,118,159,128,81,153,224,221,50,21,18,77,119,110,28,22,17,34,193,173,165,193,148,138,67,45,109,124,207,25,225,132,33,0,129,230,7,136,84,16,192,147,128,15,13,211,204,141,30,127,217,17,228,8,139,238,210,12,224,128,12,21,104,111,207,0,126,130,207,8,203,150,115,20,9,219,212,151,22,157,222,12,218,5,169,223,223,12,197,108,12,45,118,117,196,32,125,116,211,23,205,138,133,26,11,203,235,131,20,152,204,13,213,0,153,221,220,197,117,26,51,71,81,213,9,141,189,164,103,52,48,121,195,186,186,162,146,133,123,58,5,6,1,225,208,172,178,167,115,26,1,1,229,215,190,195,186,147,152,110,11,174,137,1,1,1,41,93,82,151,244,221,195,155,112,188,166,219,188,161,115,203,186,172,129,58,138,103,71,30,88,135,225,182,162,213,186,163,125,137,130,35,47,58,70,81,104,115,127,139,150,162,173,185,196,207,219,230,220,86,173,221,113,97,165,155,132,118,111,163,129,114,185,155,104,104,110,142,127,132,148,145,152,129,138,114,99,114,121,92,123,140,155,3,186,235,232,230,227,179,137,171,140,153,113,139,120,138,81,128,165,115,140,149,127,135,155,134,132,119,160,129,82,154,225,222,51,22,19,78,120,111,29,23,18,35,194,174,166,194,149,139,68,46,110,125,208,26,226,133,34,1,130,231,8,137,85,17,193,148,129,16,14,212,205,142,31,128,218,18,229,9,140,239,211,13,225,129,13,22,105,112,208,1,127,131,208,9,204,151,116,21,10,220,213,152,23,158,223,13,219,6,170,224,224,13,198,109,13,46,119,118,197,33,126,117,212,24,206,139,134,27,12,204,236,132,21,153,205,14,214,1,154,222,221,198,118,27,52,72,82,214,10,140,188,163,102,51,47,120,194,185,185,161,145,132,122,57,4,5,0,224,207,171,177,166,114,25,0,0,228,214,189,194,185,146,151,109,10,173,136,0,0,0,40,92,81,150,243,220,194,154,111,187,165,218,187,160,114,202,185,171,128,57,137,102,70,29,87,134,224,181,161,212,185,162,124,136,129,34,46,57,69,80,103,114,126,138,149,161,172,184,195,206,218,229,219,85,172,220,112,96,164,154,131,117,110,162,128,113,184,154,103,103,109,141,126,131,147,144,151,128,137,113,98,113,120,91,122,139,154,2,185,234,231,229,226,178,136,170,139,152,112,138,119,137,80,127,164,114,139,148,126,134,154,133,131,118,159,128,81,153,224,221,50,21,18,77,119,110,28,22,17,34,193,173,165,193,148,138,67,45,109,124,207,25,225,132,33,0,129,230,7,136,84,16,192,147,128,15,13,211,204,141,30,127,217,17,228,8,139,238,210,12,224,128,12,21,104,111,207,0,126,130,207,8,203,150,115,20,9,219,212,151,22,157,222,12,218,5,169,223,223,12,197,108,12,45,118,117,196,32,125,116,211,23,205,138,133,26,11,203,235,131,20,152,204,13,213,0,153,221,220,197,117,26,51,71,81,213,9,141,189,164,103,52,48,121,195,186,186,162,146,133,123,58,5,6,1,225,208,172,178,167,115,26,1,1,229,215,190,195,186,147,152,110,11,174,137,1,1,1,41,93,82,151,244,221,195,155,112,188,166,219,188,161,115,203,186,172,129,58,138,103,71,30,88,135,225,182,162,213,186,163,125,137,130,35,47,58,70,81,104,115,127,139,150,162,173,185,196,207,219,230,220,86,173,221,113,97,165,155,132,118,111,163,129,114,185,155,104,104,110,142,127,132,148,145,152,129,138,114,99,114,121,92,123,140,155,3,186,235,232,230,227,179,137,171,140,153,113,139,120,138,81,128,165,115,140,149,127,135,155,134,132,119,160,129,82,154,225,222,51,22,19,78,120,111,29,23,18,35,194,174,166,194,149,139,68,46,110,125,208,26,226,133,34,1,130,231,8,137,85,17,193,148,129,16,14,212,205,142,31,128,218,18,229,9,140,239,211,13,225,129,13,22,105,112,208,1,127,131,208,9,204,151,116,21,10,220,213,152,23,158,223,13,219,6,170,224,224,13,198,109,13,46,119,118,197,33,126,117,212,24,206,139,134,27,12,204,236,132,21,153,205,14,214,1,154,222,221,198,118,27,52,72,82,214,10]
            list_r=[197,236,209,161,80,89,255,229,228,226,227,217,204,199,165,134,112,68,255,238,222,224,223,190,100,91,0,253,239,234,227,223,208,189,187,148,225,165,123,114,56,60,120,115,197,243,250,245,234,208,245,239,255,250,243,215,254,245,246,217,132,203,162,145,103,165,195,244,217,255,241,158,212,169,174,174,45,60,75,90,105,135,150,165,180,195,210,225,240,255,255,255,255,255,141,224,255,161,144,203,216,188,154,146,207,175,160,229,200,142,153,166,187,161,175,187,183,211,171,173,156,164,148,165,142,173,215,202,113,226,255,252,249,246,217,178,229,195,187,205,174,173,182,234,192,218,172,206,215,188,181,211,177,211,148,205,165,199,194,243,240,141,76,51,175,198,184,102,113,85,96,255,255,243,255,212,198,116,113,209,215,242,56,255,204,49,66,191,255,72,181,133,52,255,195,178,37,69,245,233,198,51,195,255,43,255,61,195,255,234,22,248,176,50,63,162,176,238,50,211,180,243,59,231,200,165,77,32,249,242,198,62,191,255,43,240,35,221,250,255,86,237,196,50,96,199,210,225,64,205,185,241,60,235,215,202,89,44,228,255,210,56,214,233,74,244,90,225,255,249,220,197,56,112,155,187,231,76,198,237,210,162,81,90,256,230,229,227,228,218,205,200,166,135,113,69,256,239,223,225,224,191,101,92,1,254,240,235,228,224,209,190,188,149,226,166,124,115,57,61,121,116,198,244,251,246,235,209,246,240,256,251,244,216,255,246,247,218,133,204,163,146,104,166,196,245,218,256,242,159,213,170,175,175,46,61,76,91,106,136,151,166,181,196,211,226,241,256,256,256,256,256,142,225,256,162,145,204,217,189,155,147,208,176,161,230,201,143,154,167,188,162,176,188,184,212,172,174,157,165,149,166,143,174,216,203,114,227,256,253,250,247,218,179,230,196,188,206,175,174,183,235,193,219,173,207,216,189,182,212,178,212,149,206,166,200,195,244,241,142,77,52,176,199,185,103,114,86,97,256,256,244,256,213,199,117,114,210,216,243,57,256,205,50,67,192,256,73,182,134,53,256,196,179,38,70,246,234,199,52,196,256,44,256,62,196,256,235,23,249,177,51,64,163,177,239,51,212,181,244,60,232,201,166,78,33,250,243,199,63,192,256,44,241,36,222,251,256,87,238,197,51,97,200,211,226,65,206,186,242,61,236,216,203,90,45,229,256,211,57,215,234,75,245,91,226,256,250,221,198,57,113,156,188,232,77,197,236,209,161,80,89,255,229,228,226,227,217,204,199,165,134,112,68,255,238,222,224,223,190,100,91,0,253,239,234,227,223,208,189,187,148,225,165,123,114,56,60,120,115,197,243,250,245,234,208,245,239,255,250,243,215,254,245,246,217,132,203,162,145,103,165,195,244,217,255,241,158,212,169,174,174,45,60,75,90,105,135,150,165,180,195,210,225,240,255,255,255,255,255,141,224,255,161,144,203,216,188,154,146,207,175,160,229,200,142,153,166,187,161,175,187,183,211,171,173,156,164,148,165,142,173,215,202,113,226,255,252,249,246,217,178,229,195,187,205,174,173,182,234,192,218,172,206,215,188,181,211,177,211,148,205,165,199,194,243,240,141,76,51,175,198,184,102,113,85,96,255,255,243,255,212,198,116,113,209,215,242,56,255,204,49,66,191,255,72,181,133,52,255,195,178,37,69,245,233,198,51,195,255,43,255,61,195,255,234,22,248,176,50,63,162,176,238,50,211,180,243,59,231,200,165,77,32,249,242,198,62,191,255,43,240,35,221,250,255,86,237,196,50,96,199,210,225,64,205,185,241,60,235,215,202,89,44,228,255,210,56,214,233,74,244,90,225,255,249,220,197,56,112,155,187,231,76,197,236,209,161,80,89,255,229,228,226,227,217,204,199,165,134,112,68,255,238,222,224,223,190,100,91,0,253,239,234,227,223,208,189,187,148,225,165,123,114,56,60,120,115,197,243,250,245,234,208,245,239,255,250,243,215,254,245,246,217,132,203,162,145,103,165,195,244,217,255,241,158,212,169,174,174,45,60,75,90,105,135,150,165,180,195,210,225,240,255,255,255,255,255,141,224,255,161,144,203,216,188,154,146,207,175,160,229,200,142,153,166,187,161,175,187,183,211,171,173,156,164,148,165,142,173,215,202,113,226,255,252,249,246,217,178,229,195,187,205,174,173,182,234,192,218,172,206,215,188,181,211,177,211,148,205,165,199,194,243,240,141,76,51,175,198,184,102,113,85,96,255,255,243,255,212,198,116,113,209,215,242,56,255,204,49,66,191,255,72,181,133,52,255,195,178,37,69,245,233,198,51,195,255,43,255,61,195,255,234,22,248,176,50,63,162,176,238,50,211,180,243,59,231,200,165,77,32,249,242,198,62,191,255,43,240,35,221,250,255,86,237,196,50,96,199,210,225,64,205,185,241,60,235,215,202,89,44,228,255,210,56,214,233,74,244,90,225,255,249,220,197,56,112,155,187,231,76,198,237,210,162,81,90,256,230,229,227,228,218,205,200,166,135,113,69,256,239,223,225,224,191,101,92,1,254,240,235,228,224,209,190,188,149,226,166,124,115,57,61,121,116,198,244,251,246,235,209,246,240,256,251,244,216,255,246,247,218,133,204,163,146,104,166,196,245,218,256,242,159,213,170,175,175,46,61,76,91,106,136,151,166,181,196,211,226,241,256,256,256,256,256,142,225,256,162,145,204,217,189,155,147,208,176,161,230,201,143,154,167,188,162,176,188,184,212,172,174,157,165,149,166,143,174,216,203,114,227,256,253,250,247,218,179,230,196,188,206,175,174,183,235,193,219,173,207,216,189,182,212,178,212,149,206,166,200,195,244,241,142,77,52,176,199,185,103,114,86,97,256,256,244,256,213,199,117,114,210,216,243,57,256,205,50,67,192,256,73,182,134,53,256,196,179,38,70,246,234,199,52,196,256,44,256,62,196,256,235,23,249,177,51,64,163,177,239,51,212,181,244,60,232,201,166,78,33,250,243,199,63,192,256,44,241,36,222,251,256,87,238,197,51,97,200,211,226,65,206,186,242,61,236,216,203,90,45,229,256,211,57,215,234,75,245,91,226,256,250,221,198,57,113,156,188,232,77]
            list_b=[133,180,164,94,53,42,177,152,142,143,115,100,67,88,0,0,0,0,196,180,127,132,117,60,0,0,0,200,189,157,124,151,110,120,74,0,164,105,0,0,0,40,80,68,130,242,196,151,95,56,149,128,179,134,120,75,182,158,142,76,17,103,68,29,15,51,80,189,134,136,188,84,119,183,117,188,30,40,50,60,70,90,100,110,120,130,140,150,160,170,180,190,200,172,36,105,177,88,72,141,127,104,97,87,127,97,101,152,131,83,74,75,105,120,82,129,111,114,107,123,94,69,93,102,68,70,120,131,0,143,229,226,224,221,144,101,149,114,135,68,118,93,119,59,101,129,91,109,130,102,103,124,116,113,106,132,92,72,126,219,216,28,7,10,52,99,92,10,0,0,19,178,153,146,153,106,99,34,0,41,64,175,16,196,91,15,0,85,203,4,121,72,18,170,116,98,6,20,194,188,111,19,107,201,26,210,0,111,222,203,21,209,89,21,24,72,62,190,8,110,120,196,24,184,140,105,30,11,195,189,123,37,142,204,22,199,6,144,212,191,25,175,99,19,19,110,99,175,32,105,91,211,26,184,107,121,0,22,198,239,118,17,140,190,30,190,0,114,212,198,175,103,26,38,71,71,191,26,133,180,164,94,53,42,177,152,142,143,115,100,67,88,0,0,0,0,196,180,127,132,117,60,0,0,0,200,189,157,124,151,110,120,74,0,164,105,0,0,0,40,80,68,130,242,196,151,95,56,149,128,179,134,120,75,182,158,142,76,17,103,68,29,15,51,80,189,134,136,188,84,119,183,117,188,30,40,50,60,70,90,100,110,120,130,140,150,160,170,180,190,200,172,36,105,177,88,72,141,127,104,97,87,127,97,101,152,131,83,74,75,105,120,82,129,111,114,107,123,94,69,93,102,68,70,120,131,0,143,229,226,224,221,144,101,149,114,135,68,118,93,119,59,101,129,91,109,130,102,103,124,116,113,106,132,92,72,126,219,216,28,7,10,52,99,92,10,0,0,19,178,153,146,153,106,99,34,0,41,64,175,16,196,91,15,0,85,203,4,121,72,18,170,116,98,6,20,194,188,111,19,107,201,26,210,0,111,222,203,21,209,89,21,24,72,62,190,8,110,120,196,24,184,140,105,30,11,195,189,123,37,142,204,22,199,6,144,212,191,25,175,99,19,19,110,99,175,32,105,91,211,26,184,107,121,0,22,198,239,118,17,140,190,30,190,0,114,212,198,175,103,26,38,71,71,191,26,133,180,164,94,53,42,177,152,142,143,115,100,67,88,0,0,0,0,196,180,127,132,117,60,0,0,0,200,189,157,124,151,110,120,74,0,164,105,0,0,0,40,80,68,130,242,196,151,95,56,149,128,179,134,120,75,182,158,142,76,17,103,68,29,15,51,80,189,134,136,188,84,119,183,117,188,30,40,50,60,70,90,100,110,120,130,140,150,160,170,180,190,200,172,36,105,177,88,72,141,127,104,97,87,127,97,101,152,131,83,74,75,105,120,82,129,111,114,107,123,94,69,93,102,68,70,120,131,0,143,229,226,224,221,144,101,149,114,135,68,118,93,119,59,101,129,91,109,130,102,103,124,116,113,106,132,92,72,126,219,216,28,7,10,52,99,92,10,0,0,19,178,153,146,153,106,99,34,0,41,64,175,16,196,91,15,0,85,203,4,121,72,18,170,116,98,6,20,194,188,111,19,107,201,26,210,0,111,222,203,21,209,89,21,24,72,62,190,8,110,120,196,24,184,140,105,30,11,195,189,123,37,142,204,22,199,6,144,212,191,25,175,99,19,19,110,99,175,32,105,91,211,26,184,107,121,0,22,198,239,118,17,140,190,30,190,0,114,212,198,175,103,26,38,71,71,191,26,134,181,165,95,54,43,178,153,143,144,116,101,68,89,1,1,1,1,197,181,128,133,118,61,1,1,1,201,190,158,125,152,111,121,75,1,165,106,1,1,1,41,81,69,131,243,197,152,96,57,150,129,180,135,121,76,183,159,143,77,18,104,69,30,16,52,81,190,135,137,189,85,120,184,118,189,31,41,51,61,71,91,101,111,121,131,141,151,161,171,181,191,201,173,37,106,178,89,73,142,128,105,98,88,128,98,102,153,132,84,75,76,106,121,83,130,112,115,108,124,95,70,94,103,69,71,121,132,1,144,230,227,225,222,145,102,150,115,136,69,119,94,120,60,102,130,92,110,131,103,104,125,117,114,107,133,93,73,127,220,217,29,8,11,53,100,93,11,1,1,20,179,154,147,154,107,100,35,1,42,65,176,17,197,92,16,1,86,204,5,122,73,19,171,117,99,7,21,195,189,112,20,108,202,27,211,1,112,223,204,22,210,90,22,25,73,63,191,9,111,121,197,25,185,141,106,31,12,196,190,124,38,143,205,23,200,7,145,213,192,26,176,100,20,20,111,100,176,33,106,92,212,27,185,108,122,1,23,199,240,119,18,141,191,31,191,1,115,213,199,176,104,27,39,72,72,192,27,134,181,165,95,54,43,178,153,143,144,116,101,68,89,1,1,1,1,197,181,128,133,118,61,1,1,1,201,190,158,125,152,111,121,75,1,165,106,1,1,1,41,81,69,131,243,197,152,96,57,150,129,180,135,121,76,183,159,143,77,18,104,69,30,16,52,81,190,135,137,189,85,120,184,118,189,31,41,51,61,71,91,101,111,121,131,141,151,161,171,181,191,201,173,37,106,178,89,73,142,128,105,98,88,128,98,102,153,132,84,75,76,106,121,83,130,112,115,108,124,95,70,94,103,69,71,121,132,1,144,230,227,225,222,145,102,150,115,136,69,119,94,120,60,102,130,92,110,131,103,104,125,117,114,107,133,93,73,127,220,217,29,8,11,53,100,93,11,1,1,20,179,154,147,154,107,100,35,1,42,65,176,17,197,92,16,1,86,204,5,122,73,19,171,117,99,7,21,195,189,112,20,108,202,27,211,1,112,223,204,22,210,90,22,25,73,63,191,9,111,121,197,25,185,141,106,31,12,196,190,124,38,143,205,23,200,7,145,213,192,26,176,100,20,20,111,100,176,33,106,92,212,27,185,108,122,1,23,199,240,119,18,141,191,31,191,1,115,213,199,176,104,27,39,72,72,192,27]
            list_result=['mild','fair','fair','dark','dark','dark','fair','fair','fair','mild','fair','mild','mild','mild','dark','dark','dark','dark','fair','fair','mild','mild','mild','mild','dark','dark','dark','fair','fair','fair','fair','mild','fair','dark','dark','dark','fair','mild','dark','dark','dark','dark','dark','dark','mild','fair','fair','fair','mild','mild','fair','fair','fair','fair','fair','mild','fair','fair','fair','mild','dark','mild','dark','dark','dark','dark','mild','fair','mild','mild','fair','dark','mild','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','mild','mild','mild','mild','fair','fair','fair','fair','fair','fair','dark','mild','fair','mild','dark','fair','fair','fair','mild','fair','fair','mild','mild','fair','fair','dark','dark','dark','fair','mild','fair','fair','mild','mild','mild','fair','mild','dark','mild','mild','dark','mild','fair','mild','dark','fair','fair','fair','fair','fair','fair','mild','fair','mild','fair','mild','fair','fair','mild','mild','fair','fair','mild','mild','fair','fair','mild','fair','fair','fair','mild','fair','mild','dark','fair','fair','fair','dark','dark','dark','mild','mild','mild','dark','dark','dark','dark','fair','fair','fair','fair','mild','mild','dark','dark','mild','mild','fair','dark','fair','mild','dark','dark','mild','fair','dark','mild','mild','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','fair','mild','dark','dark','mild','mild','fair','dark','mild','fair','fair','dark','dark','dark','fair','fair','fair','mild','fair','mild','mild','mild','dark','dark','dark','dark','fair','fair','mild','mild','mild','mild','dark','dark','dark','fair','fair','fair','fair','mild','fair','dark','dark','dark','fair','mild','dark','dark','dark','dark','dark','dark','mild','fair','fair','fair','mild','mild','fair','fair','fair','fair','fair','mild','fair','fair','fair','mild','dark','mild','dark','dark','dark','dark','mild','fair','mild','mild','fair','dark','mild','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','mild','mild','mild','mild','fair','fair','fair','fair','fair','fair','dark','mild','fair','mild','dark','fair','fair','fair','mild','fair','fair','mild','mild','fair','fair','dark','dark','dark','fair','mild','fair','fair','mild','mild','mild','fair','mild','dark','mild','mild','dark','mild','fair','mild','dark','fair','fair','fair','fair','fair','fair','mild','fair','mild','fair','mild','fair','fair','mild','mild','fair','fair','mild','mild','fair','fair','mild','fair','fair','fair','mild','fair','mild','dark','fair','fair','fair','dark','dark','dark','mild','mild','mild','dark','dark','dark','dark','fair','fair','fair','fair','mild','mild','dark','dark','mild','mild','fair','dark','fair','mild','dark','dark','mild','fair','dark','mild','mild','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','fair','mild','dark','dark','mild','mild','fair','dark','mild','fair','fair','dark','dark','dark','fair','fair','fair','mild','fair','mild','mild','mild','dark','dark','dark','dark','fair','fair','mild','mild','mild','mild','dark','dark','dark','fair','fair','fair','fair','mild','fair','dark','dark','dark','fair','mild','dark','dark','dark','dark','dark','dark','mild','fair','fair','fair','mild','mild','fair','fair','fair','fair','fair','mild','fair','fair','fair','mild','dark','mild','dark','dark','dark','dark','mild','fair','mild','mild','fair','dark','mild','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','mild','mild','mild','mild','fair','fair','fair','fair','fair','fair','dark','mild','fair','mild','dark','fair','fair','fair','mild','fair','fair','mild','mild','fair','fair','dark','dark','dark','fair','mild','fair','fair','mild','mild','mild','fair','mild','dark','mild','mild','dark','mild','fair','mild','dark','fair','fair','fair','fair','fair','fair','mild','fair','mild','fair','mild','fair','fair','mild','mild','fair','fair','mild','mild','fair','fair','mild','fair','fair','fair','mild','fair','mild','dark','fair','fair','fair','dark','dark','dark','mild','mild','mild','dark','dark','dark','dark','fair','fair','fair','fair','mild','mild','dark','dark','mild','mild','fair','dark','fair','mild','dark','dark','mild','fair','dark','mild','mild','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','fair','mild','dark','dark','mild','mild','fair','dark','mild','fair','fair','dark','dark','dark','fair','fair','fair','mild','fair','mild','mild','mild','dark','dark','dark','dark','fair','fair','mild','mild','mild','mild','dark','dark','dark','fair','fair','fair','fair','mild','fair','dark','dark','dark','fair','mild','dark','dark','dark','dark','dark','dark','mild','fair','fair','fair','mild','mild','fair','fair','fair','fair','fair','mild','fair','fair','fair','mild','dark','mild','dark','dark','dark','dark','mild','fair','mild','mild','fair','dark','mild','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','mild','mild','mild','mild','fair','fair','fair','fair','fair','fair','dark','mild','fair','mild','dark','fair','fair','fair','mild','fair','fair','mild','mild','fair','fair','dark','dark','dark','fair','mild','fair','fair','mild','mild','mild','fair','mild','dark','mild','mild','dark','mild','fair','mild','dark','fair','fair','fair','fair','fair','fair','mild','fair','mild','fair','mild','fair','fair','mild','mild','fair','fair','mild','mild','fair','fair','mild','fair','fair','fair','mild','fair','mild','dark','fair','fair','fair','dark','dark','dark','mild','mild','mild','dark','dark','dark','dark','fair','fair','fair','fair','mild','mild','dark','dark','mild','mild','fair','dark','fair','mild','dark','dark','mild','fair','dark','mild','mild','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','fair','mild','dark','dark','mild','mild','fair','dark','mild','fair','fair','dark','dark','dark','fair','fair','fair','mild','fair','mild','mild','mild','dark','dark','dark','dark','fair','fair','mild','mild','mild','mild','dark','dark','dark','fair','fair','fair','fair','mild','fair','dark','dark','dark','fair','mild','dark','dark','dark','dark','dark','dark','mild','fair','fair','fair','mild','mild','fair','fair','fair','fair','fair','mild','fair','fair','fair','mild','dark','mild','dark','dark','dark','dark','mild','fair','mild','mild','fair','dark','mild','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','dark','mild','mild','mild','mild','fair','fair','fair','fair','fair','fair','dark','mild','fair','mild','dark','fair','fair','fair','mild','fair','fair','mild','mild','fair','fair','dark','dark','dark','fair','mild','fair','fair','mild','mild','mild','fair','mild','dark','mild','mild','dark','mild','fair','mild','dark','fair','fair','fair','fair','fair','fair','mild','fair','mild','fair','mild','fair','fair','mild','mild','fair','fair','mild','mild','fair','fair','mild','fair','fair','fair','mild','fair','mild','dark','fair','fair','fair','dark','dark','dark','mild','mild','mild','dark','dark','dark','dark','fair','fair','fair','fair','mild','mild','dark','dark','mild','mild','fair','dark','fair','mild','dark','dark','mild','fair','dark','mild','mild','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','dark','fair','mild','dark','dark','mild','mild','fair','dark','mild','mild','fair','dark','fair','mild','mild','dark','dark','fair','fair','mild','dark','mild','fair','dark','fair','dark','mild','fair','fair','fair','mild','dark','dark','mild','mild','fair','dark']
            df=pd.DataFrame({'R':list_r,'G':list_g,'B':list_b,'Result':list_result})
    
            df.shape

            df.head()

            df["Result"].value_counts()

            "Mild".title()

            result_list=list(df["Result"].values)
            for i in range(len(result_list)):
                result_list[i]=result_list[i].title()

            df["Result"]=result_list

            df["Result"].value_counts()

            df.drop_duplicates(inplace=True)
            df.shape

            df.head(20)

            # Training Model using Decision Tree

            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split as tts
            from sklearn.neighbors import KNeighborsClassifier as KNN
            from sklearn.metrics import accuracy_score

            x=df[["R","G","B"]]
            y=df[["Result"]]

            train_x,test_x,train_y,test_y=tts(x,y,random_state=42,stratify=y,test_size=0.25)

            from sklearn.tree import DecisionTreeClassifier

            dt_model=DecisionTreeClassifier(random_state=42)
            dt_model.fit(train_x,train_y)

            train_predict=dt_model.predict(train_x)
            accuracy_score(train_predict,train_y)

            test_predict=dt_model.predict(test_x)
            accuracy_score(test_predict,test_y)

            test_x

            type(test_x)

            df.head()

            img_df=pd.DataFrame({"R":rgb[0],"G":rgb[1],"B":rgb[2]},index=[0])

            img_df

            dataset_pred=list(dt_model.predict(img_df))[0]
            list(dt_model.predict(img_df))[0]

            # Prediction using Intensity

            intensity=(0.2126*int(img_df["R"]))+(0.7152*int(img_df["G"]))+(0.0722*int(img_df["B"]))
                
            intensity_pred=""
            if intensity<=115:
                print("Dark")
                intensity_pred="Dark"
            elif intensity>115 and intensity<=150:
                print("Mild")
                intensity_pred="Mild"
            else:
                print("Fair")
                intensity_pred="Fair"

            intensity

            df[df["R"]==rgb[0]]

            # Training Model using KNN

            knn=KNN(n_neighbors=10)
            knn.fit(train_x,train_y)
            knn_pred=list(knn.predict(img_df))[0]
            list(knn.predict(img_df))[0]

            # Final Prediction using Ensembling

            if intensity>=155:
                final_pred="Fair"
            elif dataset_pred==intensity_pred==knn_pred:
                final_pred=dataset_pred
            elif dataset_pred==intensity_pred:
                final_pred=dataset_pred
            elif intensity_pred==knn_pred:
                final_pred=intensity_pred
            elif dataset_pred==knn_pred:
                final_pred=dataset_pred
            elif intensity<=120:
                final_pred="Dark"
            else:
                final_pred="Mild"

            print(final_pred)


            #img1 = Image.Objects.get(id=id1)
            #print(img1.image.path)
            # Get the current instance object to display in the template
            img_obj = form.instance
            return render(request, 'skintone_app/upload.html', {'form': form, 'img_obj': img_obj, 'prediction': final_pred})
    else:
        form = ImageForm()
    return render(request, 'skintone_app/upload.html', {'form': form})




           