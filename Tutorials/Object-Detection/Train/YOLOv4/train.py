from xtreme_vision.Detection.Custom import Train_Custom_Detector

model = Train_Custom_Detector()
model.Use_YOLOv4(classes_path = 'classes.names',
                 input_size = 608,
                 batch_size = 4)

model.load_data(train_annot_path = 'training.txt',
                train_img_dir = './',
                val_annot_path = 'validation.txt',
                val_img_dir = './',
                weights_path = None)

model.train(epochs=1,
            lr = 0.001)


#After Training, Load the Model Like this

from xtreme_vision.Detection import Object_Detection
model = Object_Detection()
model.Use_YOLOv4(weights_path = 'path-to-the-trained-weights', 
		 classes_path = 'path-to-the-classes-file')

model.Detect_From_Image(input_path='kite.jpg',
                        output_path='./out.jpg')

from PIL import Image
Image.open('out.jpg')