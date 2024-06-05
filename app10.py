import streamlit as st
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
import io
from PIL import ImageStat
from streamlit_cropper import st_cropper
import io

st.markdown("""
<style>
    body {
        font-size:20px !important;
    }
</style>
""", unsafe_allow_html=True)

def crop_image(image, left, top, right, bottom):
    return image.crop((left, top, right, bottom))

def add_salt_and_pepper(image_array, amount):
    row, col, _ = image_array.shape
    s_vs_p = 0.5
    out = np.copy(image_array)
    num_salt = np.ceil(amount * image_array.size * s_vs_p)
    num_pepper = np.ceil(amount * image_array.size * (1.0 - s_vs_p))

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
    out[coords[0], coords[1], :] = 1

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
    out[coords[0], coords[1], :] = 0
    return out

def add_gaussian_noise(image_array, mean=0, var=0.1):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image_array.shape)
    noisy_image = image_array + gauss
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def apply_filter(image, filter_type):
    return image.filter(filter_type)

def apply_canny(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def apply_sobel(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return np.uint8(sobel)

def apply_felzenszwalb(image, scale):
    from skimage.segmentation import felzenszwalb
    img = np.array(image.convert('RGB'))
    segments = felzenszwalb(img, scale=scale)
    return segments

def apply_gaussian_adaptive_thresholding(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh



def region_growing(img, seed):
    height, width = img.shape
    visited = np.zeros_like(img, dtype=np.uint8)
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    stack = [seed]

    while len(stack) > 0:
        s = stack.pop()
        x, y = s

        if np.abs(int(img[seed]) - int(img[x, y])) > 20:  # intensity difference threshold
            continue

        visited[x, y] = 255

        for direction in range(8):
            nx, ny = x + dx[direction], y + dy[direction]
            if nx >= 0 and ny >= 0 and nx < height and ny < width:
                if visited[nx, ny] == 0:
                    stack.append((nx, ny))

    return visited

def extract_features(roi):
    gray_roi = cv2.cvtColor(np.array(roi), cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_roi, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    texture_features = {
        'Contrast': contrast.mean(),
        'Dissimilarity': dissimilarity.mean(),
        'Homogeneity': homogeneity.mean(),
        'Energy': energy.mean(),
        'Correlation': correlation.mean()
    }
    area = np.count_nonzero(gray_roi)
    contours, _ = cv2.findContours(cv2.convertScaleAbs(gray_roi), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    aspect_ratio = roi.width / roi.height if roi.height != 0 else 0
    entropy = shannon_entropy(gray_roi)
    skewness = skew(gray_roi.ravel())
    kurt = kurtosis(gray_roi.ravel())
    geometrical_statistical_features = {
        'Area': area,
        'Perimeter': perimeter,
        'Aspect Ratio': aspect_ratio,
        'Entropy': entropy,
        'Skewness': skewness,
        'Kurtosis': kurt
    }
    return texture_features, geometrical_statistical_features

def display_features(texture_features, geometrical_statistical_features):
    st.write("### Texture Features")
    for feature, value in texture_features.items():
        st.write(f"{feature}: {value:.4f}")
    
    st.write("### Geometrical and Statistical Features")
    for feature, value in geometrical_statistical_features.items():
        st.write(f"{feature}: {value:.4f}")

def main():
    
    st.title('Defect Detection')

    tasks = ['Interactive Defect Detection', 'Visualization']
    choice = st.sidebar.selectbox('Select Task', tasks)
    

    if choice == 'Visualization':
        st.subheader('Image Visualization')
        image_file = st.file_uploader('Upload', type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            img_array = np.array(our_image)
            

            process_choice = st.sidebar.selectbox("Choose a process", ["None", "Crop", "Flip Horizontal", "Flip Vertical", "Add Noise", "Apply Filter", "Edge Detection", "Segmentation","Image Properties"])

            if process_choice not in ["None" ,"Image Properties" ,"Crop"]:
                col1, col2 = st.columns(2)
                col1.markdown('Original Image')
                col1.image(our_image, width=325)
                col2.markdown('Modified Image')
                enhance_type = st.sidebar.radio("Enhance type", ['Original', 'Gray-scale', 'Contrast', 'Brightness', 'Blurring', 'Sharpness'])

                if enhance_type == 'Gray-scale':
                    img = np.array(our_image.convert('RGB'))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    col2.image(gray, width=325)
                elif enhance_type == 'Contrast':
                    rate = st.sidebar.slider("Contrast", 0.5, 6.0)
                    enhancer = ImageEnhance.Contrast(our_image)
                    enhanced_img = enhancer.enhance(rate)
                    col2.image(enhanced_img, width=325)
                elif enhance_type == 'Brightness':
                    rate = st.sidebar.slider("Brightness", 0.0, 8.0)
                    enhancer = ImageEnhance.Brightness(our_image)
                    enhanced_img = enhancer.enhance(rate)
                    col2.image(enhanced_img, width=325)
                elif enhance_type == 'Blurring':
                    rate = st.sidebar.slider("Blurring", 0.0, 7.0)
                    blurred_img = cv2.GaussianBlur(np.array(our_image), (15, 15), rate)
                    col2.image(blurred_img, width=325)
                elif enhance_type == 'Sharpness':
                    rate = st.sidebar.slider("Sharpness", 0.0, 14.0)
                    enhancer = ImageEnhance.Sharpness(our_image)
                    enhanced_img = enhancer.enhance(rate)
                    col2.image(enhanced_img, width=325)
                else:
                    col2.image(our_image, width=325)

            
            if process_choice == "Crop":
                st.markdown('Crop Image')
                # from streamlit_cropper import st_cropper
                realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
                box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
                aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
                aspect_dict = {
                    "1:1": (1, 1),
                    "16:9": (16, 9),
                    "4:3": (4, 3),
                    "2:3": (2, 3),
                    "Free": None
                }
                aspect_ratio = aspect_dict[aspect_choice]
                
                if not realtime_update:
                    st.write("Double click to save crop")
                
                cropped_img = st_cropper(our_image, realtime_update=realtime_update, box_color=box_color, aspect_ratio=aspect_ratio)
                    
                
                buf = io.BytesIO()
                cropped_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                
                    
                col1, col2 = st.columns(2)
                with col2:
                    st.write("Preview")
                    _ = cropped_img.thumbnail((150,150))
                    st.image(cropped_img, caption='Cropped Image', use_column_width=True)
                    st.download_button(
                        label="Download cropped image",
                        data=byte_im,
                        file_name="cropped_image.png",
                        mime="image/png"
                    )

                with col1:
                    st.markdown('Features')
                    if cropped_img.size[0] > 0 and cropped_img.size[1] > 0:
                        texture_features, geometrical_statistical_features = extract_features(cropped_img)
                        display_features(texture_features, geometrical_statistical_features)
                    else:
                        st.write("Selected region is empty or invalid.")
                
                
            elif process_choice == "None":
                st.markdown('Original Image')
                st.image(our_image, width=325)
                st.write("No process selected")
            elif process_choice == "Flip Horizontal":
                flipped_image = our_image.transpose(Image.FLIP_LEFT_RIGHT)
                col2.image(flipped_image, width=325)
            elif process_choice == "Flip Vertical":
                flipped_image = our_image.transpose(Image.FLIP_TOP_BOTTOM)
                col2.image(flipped_image, width=325)
            elif process_choice == "Add Noise":
                noise_type = st.sidebar.radio("Noise type", ["Salt and Pepper", "Gaussian"])
                if noise_type == "Salt and Pepper":
                    noise_amount = st.sidebar.slider("Amount", 0.01, 0.1, 0.05)
                    noisy_image = add_salt_and_pepper(np.array(our_image.convert('RGB')), noise_amount)
                    col2.image(noisy_image, width=325)
                elif noise_type == "Gaussian":
                    noisy_image = add_gaussian_noise(np.array(our_image.convert('RGB')))
                    col2.image(noisy_image, width=325)
            elif process_choice == "Apply Filter":
                filter_type = st.sidebar.radio("Filter type", ["Average", "Median", "Min", "Max", "Mode", "Gaussian","Smoothing"])
                if filter_type == "Average":
                    filtered_image = apply_filter(our_image, ImageFilter.BoxBlur(3))
                elif filter_type == "Median":
                    filtered_image = apply_filter(our_image, ImageFilter.MedianFilter(size=3))
                elif filter_type == "Min":
                    filtered_image = apply_filter(our_image, ImageFilter.MinFilter(size=3))
                elif filter_type == "Max":
                    filtered_image = apply_filter(our_image, ImageFilter.MaxFilter(size=3))
                elif filter_type == "Mode":
                    filtered_image = apply_filter(our_image, ImageFilter.ModeFilter(size=3))
                elif filter_type == "Gaussian":
                    filtered_image = apply_filter(our_image, ImageFilter.GaussianBlur(radius=3))
                elif filter_type == "Smoothing":
                    blur_amount = st.sidebar.slider("Blur amount", 1, 51, 5, step=2)  # The blur amount must be an odd number
                    filtered_image = cv2.GaussianBlur(np.array(our_image), (blur_amount, blur_amount), 0)
                
                col2.image(filtered_image, width=325)
            elif process_choice == "Edge Detection":
                edge_type = st.sidebar.radio("Edge detection type", ["Canny", "Sobel"])
                if edge_type == "Canny":
                    edge_image = apply_canny(our_image)
                elif edge_type == "Sobel":
                    edge_image = apply_sobel(our_image)
                col2.image(edge_image, width=325)
            
            elif process_choice == "Segmentation":
                seg_type = st.sidebar.radio("Segmentation type", ["Edge based","Morphological based","Region Growing"])
                if seg_type == "Edge based":
                    edge_image = apply_canny(our_image)
                    col2.image(edge_image, width=325)
                elif seg_type == "Morphological based":
                    operation = st.sidebar.radio("Choose an operation", ["Erosion", "Dilation", "Opening", "Closing", "Thickening"])
                    kernel_size = st.sidebar.slider("Kernel size", 1, 7, 3, step=2)  # The kernel size must be an odd number
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                    if operation == "Erosion":
                        morph_image = cv2.erode(np.array(our_image), kernel, iterations=1)
                    elif operation == "Dilation":
                        morph_image = cv2.dilate(np.array(our_image), kernel, iterations=1)
                    elif operation == "Opening":
                        morph_image = cv2.morphologyEx(np.array(our_image), cv2.MORPH_OPEN, kernel)
                    elif operation == "Closing":
                        morph_image = cv2.morphologyEx(np.array(our_image), cv2.MORPH_CLOSE, kernel)
                    elif operation == "Thickening":
                        morph_image = cv2.dilate(cv2.erode(np.array(our_image), kernel, iterations=1), kernel, iterations=1)
                    col2.image(morph_image, width=325)
                elif seg_type == "Region Growing":
                    seed = (300, 300)  # You can change this to let the user select the seed point
                    segmented_image = region_growing(np.array(our_image.convert("L")), seed)
                    col2.image(segmented_image, width=325)
            elif process_choice == "Image Properties":
                st.sidebar.write("Image size: ", our_image.size)
                st.sidebar.write("Image mode: ", our_image.mode)
                st.sidebar.write("Image format: ", our_image.format)
                # Additional properties
                st.sidebar.write("Image width: ", our_image.width)
                st.sidebar.write("Image height: ", our_image.height)
                st.sidebar.write("Image bands: ", our_image.getbands())
                # Image statistics
                stats = ImageStat.Stat(our_image)
                st.sidebar.write("Image mean: ", stats.mean)
                st.sidebar.write("Image median: ", stats.median)
                st.sidebar.write("Image variance: ", stats.var)

            # elif process_choice == "Smoothing":
            #     blur_amount = st.sidebar.slider("Blur amount", 1, 51, 5, step=2)  # The blur amount must be an odd number
            #     smoothed_image = cv2.GaussianBlur(np.array(our_image), (blur_amount, blur_amount), 0)
            #     col2.image(smoothed_image, width=325)

    elif choice == 'Interactive Defect Detection':
        st.subheader('LOP or Porosity detection')
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
        loaded_model = load_model("efficientnet_model_compressed.h5")
        def predict_image_class(img_path):
            img = image.load_img(img_path, target_size=(640, 640))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            predictions = loaded_model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            return predicted_class

        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            # st.write("")
            # st.write("Classifying...")

            # predicted_class = predict_image_class(uploaded_file)
            # class_names = ["lop", "porosity"]
            # st.write(f"Predicted State: {class_names[predicted_class]}")

            st.markdown('Crop Image')
            # from streamlit_cropper import st_cropper
            realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
            box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
            aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
            aspect_dict = {
                "1:1": (1, 1),
                "16:9": (16, 9),
                "4:3": (4, 3),
                "2:3": (2, 3),
                "Free": None
            }
            aspect_ratio = aspect_dict[aspect_choice]
                        
            if not realtime_update:
                st.write("Double click to save crop")
            
            uploaded_image = Image.open(uploaded_file)

            cropped_img = st_cropper(uploaded_image, realtime_update=realtime_update, box_color=box_color, aspect_ratio=aspect_ratio)
                            
                        
            buf = io.BytesIO()
            cropped_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
                        
            # st.download_button(
            #     label="Download cropped image",
            #     data=byte_im,
            #     file_name="cropped_image.png",
            #     mime="image/png"
            # )
                            
            col1, col2 = st.columns(2)
            with col2:
                st.write("Preview")
                _ = cropped_img.thumbnail((150,150))
                st.image(cropped_img, caption='Cropped Image', use_column_width=True)

            with col1:
                st.markdown('Features')
                if cropped_img.size[0] > 0 and cropped_img.size[1] > 0:
                    texture_features, geometrical_statistical_features = extract_features(cropped_img)
                    display_features(texture_features, geometrical_statistical_features)
                    st.markdown("")
                    st.download_button(
                        label="Download cropped image",
                        data=byte_im,
                        file_name="cropped_image.png",
                        mime="image/png"
                    )
                else:
                    st.write("Selected region is empty or invalid.")
                

            # loaded_model = load_model("efficientnet_model_compressed.h5")
                            
            def predict_image_class_cropped_image(img_path):
                img = image.load_img(img_path, target_size=(640, 640))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                predictions = loaded_model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]
                return predicted_class

            def predict_image_class_original_image(img_path):
                img = image.load_img(img_path, target_size=(640, 640))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                predictions = loaded_model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]
                return predicted_class
                        
            with col2:
                if cropped_img is not None:

                    if uploaded_image is not None:
                        col2.write("")
                        col2.write("Original Image Prediction...")

                        image_stream1 = io.BytesIO()
                        uploaded_image.save(image_stream1, format='JPEG')
                        image_stream1.seek(0)
                                
                        predicted_class = predict_image_class_original_image(image_stream1)
                        class_names = ["lop", "porosity"]
                        col2.write(f"Predicted State: {class_names[predicted_class]}")

                    col2.write("")
                    col2.write("Cropped Area Prediction...")

                    image_stream2 = io.BytesIO()
                    cropped_img.save(image_stream2, format='JPEG')
                    image_stream2.seek(0)

                    predicted_class = predict_image_class_cropped_image(image_stream2)
                    class_names = ["lop", "porosity"]
                    col2.write(f"Predicted State: {class_names[predicted_class]}")

if __name__ == '__main__':
    main()
