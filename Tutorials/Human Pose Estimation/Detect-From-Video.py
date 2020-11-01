from xtreme_vision.Estimation import Pose_Estimation

model = Pose_Estimation()
model.Use_CenterNet()
model.Detect_From_Video(input_path = 'video.avi', 
                        output_path = 'output.mp4')