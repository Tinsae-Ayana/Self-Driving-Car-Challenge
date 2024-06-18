import utils
import numpy as np


# use he model and perform segmnation 
def segment_image(model, input_img):
   model.eval()
   logit  = model(input_img)["out"]
   pred   = logit.softmax(dim=1)
   pred   = pred.argmax(1).squeeze(dim=0)
   return pred


# produce the bev for video sequence
def main():

    # loading segmentation model
    seg_model = utils.load_seg_model()
    print("semantic segmentation model loaded................................................")

    # loading bev model
    bev_model = utils.load_bev_model()
    print("bird eye view model loaded........................................................")

    # load test image and segment it
    img = utils.load_image("src/test_image3.png")
    print(f"image: {img.shape}")
    seg_input = img.unsqueeze(0)
    seg_input = segment_image(model=seg_model, input_img=seg_input)
    seg_input = utils.apply_color_map_seg(seg_input.squeeze())
    print(f"segmented: {seg_input.shape}")
    # convert the segmented image to the format of bev input
    bev_input = utils.one_hot_encode_image_op(image=seg_input.numpy())
    
    # get the bev
    pred = bev_model.predict(np.expand_dims(bev_input, axis=0)).squeeze()
    output = utils.one_hot_decode_image(pred)
    print(f"output: {output.shape}")
    utils.visualize_image(img.permute(1,2,0), seg_input, output)

if __name__ == "__main__":
    main()


