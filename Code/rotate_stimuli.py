import os
from PIL import Image

# Function to rotate an image 180 degrees
def rotate_image(image_path):
    with Image.open(image_path) as img:
        return img.rotate(180)

# Main function to process the folders and images
def rotate_images_in_folders(src_base_folder, dst_base_folder):
    # Ensure the destination base folder exists
    os.makedirs(dst_base_folder, exist_ok=True)
    
    # Get the list of subfolders in the source base folder
    subfolders = [f.path for f in os.scandir(src_base_folder) if f.is_dir()]

    for subfolder in subfolders:
        # Create the corresponding subfolder in the destination base folder
        subfolder_name = os.path.basename(subfolder)
        rotated_subfolder = os.path.join(dst_base_folder, subfolder_name + '_rotated')
        os.makedirs(rotated_subfolder, exist_ok=True)

        # Iterate through all images in the current source subfolder
        for image_name in os.listdir(subfolder):
            image_path = os.path.join(subfolder, image_name)
            if os.path.isfile(image_path):
                # Rotate the image
                rotated_image = rotate_image(image_path)
                # Save the rotated image to the new subfolder
                rotated_image_path = os.path.join(rotated_subfolder, image_name)
                rotated_image.save(rotated_image_path)

if __name__ == '__main__':
    # Define the source and destination base folders
    src_base_folder = r"C:\Users\HP\Desktop\Studies\FaceRec seminar\CompositeEffectExperiment2\Stimuli"
    dst_base_folder = r"C:\Users\HP\Desktop\Studies\FaceRec seminar\CompositeEffectExperiment2\Upside_Stimuli"

    # Run the function
    rotate_images_in_folders(src_base_folder, dst_base_folder)
