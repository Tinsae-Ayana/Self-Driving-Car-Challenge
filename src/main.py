import utils
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image

# use he model and perform segmnation 
def segment_image(model, input_img):
   model.eval()
   logit  = model(input_img)["out"]
   pred   = logit.softmax(dim=1)
   pred   = pred.argmax(1).squeeze(dim=0)
   return pred

# resize frame
def resize(frame):
    frame_pil = Image.fromarray(frame)
    img_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])
    return img_transform(frame_pil)

# produce the bev for video sequence
def main():

    # loading segmentation model
    seg_model = utils.load_seg_model()
    print("semantic segmentation model loaded................................................")

    # loading bev model
    bev_model = utils.load_bev_model()
    print("bird eye view model loaded........................................................")

    # load the video
    video_path  = "assets/sample.mp4"
    output_path = "assets/output.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open the video")
        exit()

    # setup the output vidoe
    frame_width  = 512
    frame_height = 256
    fps          = int(cap.get(cv2.CAP_PROP_FPS)) 
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
    out          = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))   
    
    print("Processing...................................................")
    while True:
       
        ret, frame = cap.read()
        if not ret:
            break

        print(f"image: {frame.shape}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg_input = resize(frame).unsqueeze(0)
        seg_input = segment_image(model=seg_model, input_img=seg_input)
        seg_input = utils.apply_color_map_seg(seg_input.squeeze())
        bev_input = utils.one_hot_encode_image_op(image=seg_input.numpy())
        
        # get the bev
        pred = bev_model.predict(np.expand_dims(bev_input, axis=0)).squeeze()
        output = utils.one_hot_decode_image(pred)
       
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        out.write(output)
    print("Done!")

    cap.release()
    out.release()
   

if __name__ == "__main__":
    main()



