import os

def get_data(path:str="data"):
    # extract all the images
    images = []
    for image in os.listdir(f"{path}\images"):
        images.append(image)
    # extract the captions
    captions = []
    for caption in open(f"{path}\captions.txt",'r').readlines()[1:]:
        captions.append(caption.split(","))
        return zip(images, captions)



if __name__ == '__main__':
    all_data = get_data()
    print(all_data)
