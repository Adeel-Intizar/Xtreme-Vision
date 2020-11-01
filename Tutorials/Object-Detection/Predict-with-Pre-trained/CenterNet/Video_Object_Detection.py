from xtreme_vision.Detection import Object_Detection

model = Object_Detection()
model.Use_CenterNet()
model.Detect_From_Video(input_path = 'road.mp4', 
                        output_path='./cetnernet.mp4')