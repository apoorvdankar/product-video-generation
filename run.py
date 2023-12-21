import torch
from PIL import Image
import torch
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import imageio
import argparse

SIZE = 512
RESIZE_FACTOR = 0.25

def shift_image(image, x, y, constant_value=255):
    shifted_image = np.roll(image, (y, x), axis=(0, 1))
    if x>=0:
        shifted_image[:, :x, :] = constant_value
    else:
        shifted_image[:, x:, :] = constant_value
    if y>=0:
        shifted_image[:y, :, :] = constant_value
    else:
        shifted_image[y:, :, :] = constant_value
    return Image.fromarray(np.uint8(shifted_image))


def generate_image(image_path, prompt, image_output_path, device):
    # Loading the Runway fine tuned diffusion model for inpainting
    pipe_runway = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    image = Image.open(image_path)
    WIDTH, HEIGHT = image.size

    #Location of the product in the generated image
    LOC_X, LOC_Y = 100, 100 

    image_resized = image.resize((int(WIDTH * RESIZE_FACTOR), int(HEIGHT * RESIZE_FACTOR)), Image.LANCZOS)
    WIDTH_NEW, HEIGHT_NEW = image_resized.size

    # Adding white background to the resized image
    image_white_background_array = np.ones((SIZE, SIZE, 3)) * 255
    image_white_background_array[LOC_Y:LOC_Y+HEIGHT_NEW, LOC_X:LOC_X+WIDTH_NEW, :] = np.array(image_resized)
    image_white_background = Image.fromarray(np.uint8(image_white_background_array))

    # Creating the mask to be passed to the inpainting model
    # Currently using pixel values to segment, further work can include segmentation model
    mask_array = np.where(image_white_background_array[:,:,2] < 250, 0, 255)
    mask_array = np.tile(mask_array[:, :, np.newaxis], 3)
    mask_image = Image.fromarray(np.uint8(mask_array))

    # Boolean mask and reverse mask to retain the actual image in the generated image.
    # This is needed to retain high quality image as even with inpainting mask, there can be some degradation
    mask_bool = np.where(mask_array == 0, 0, 1)
    reverse_mask_bool = 1 - mask_bool

    new_image_runway = pipe_runway(prompt=prompt,
                            image=image_white_background,
                            mask_image=mask_image,
                            num_inference_steps=50).images[0]

    # Retain the actual image in the generated image.
    image_output = new_image_runway*mask_bool + image_white_background*reverse_mask_bool
    image_output = Image.fromarray(np.uint8(image_output))

    image_output.save(image_output_path)
    return image_output


def generate_video(image, prompt, video_output_path, device):
    # During experiments observed that stabilityai model works better for this
    pipe_stability = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to(device)

    # This is to specify the path of the objects motion in the video
    # This can also be made as one of the inputs to the function to let users provide the path
    shifts = [(5,10)]*10 +[(10,5)]*10

    frames = [image]
    for i in range(len(shifts)):
        x, y = shifts[i]
        image_array = np.array(image)
        #Shift the image to create a motion in the video
        shifted_image = shift_image(image_array, x, y, constant_value=255)
        shifted_mask = shift_image(np.zeros_like(image_array), x, y, constant_value=255)

        shift_mask_bool = np.where(np.array(shifted_mask) == 0, 0, 1)
        shift_reverse_mask_bool = 1 - shift_mask_bool

        image_out = pipe_stability(prompt=prompt, image=shifted_image, mask_image=shifted_mask).images[0]

        # Using only the newly generated part and keeping the rest same as original image
        # This helps in reducing the error propagation throught the decoding and encoding of LDM
        image_out_retained = Image.fromarray(
            np.uint8(image_out*shift_mask_bool + shifted_image*shift_reverse_mask_bool)
        )
        frames.append(image_out_retained)
        image = image_out_retained

    # Specify the frames per second (fps)
    fps = 10

    # Convert PIL images to numpy arrays
    image_array = [np.array(img) for img in frames]

    # Save the images as a video
    imageio.mimsave(video_output_path, image_array, fps=fps)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="./input/example4.jpg", type=str)
    parser.add_argument("--text_prompt", default="chair in an office", type=str)
    parser.add_argument("--image_output", default="./output/gen_image3.jpg", type=str)
    parser.add_argument("--video_output", default="./output/gen_video3.mp4", type=str)
    parser.add_argument("--device", default="mps", type=str)
    args = parser.parse_args()

    image_output = generate_image(image_path=args.image, 
                                  prompt=args.text_prompt, 
                                  image_output_path=args.image_output, 
                                  device=args.device)
    
    if args.video_output:
        generate_video(image=image_output, 
                       prompt=args.text_prompt, 
                       video_output_path=args.video_output, 
                       device=args.device
                       )


if __name__ == "__main__":
    main()