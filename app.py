from more_itertools import sort_together
from cmath import cos
from sklearn.cluster import KMeans
import webcolors
import matplotlib
import matplotlib.pyplot as plt
import cv2
from colormap import rgb2hex
from fastai.vision.all import *
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import webbrowser
import gc


@st.cache(allow_output_mutation=True)
def load_image(image_file):
    img = Image.open(image_file)
    return img


st. set_page_config(layout="wide")
st.title('Fashion Recommendation System')
st.markdown("--------------------")

# Getting the user's input image
image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
if image_file is not None:
    file_details = {"Filename": image_file.name,
                    "FileType": image_file.type, "FileSize": image_file.size}
    file_button = st.button("Show File Details",
                            help="Click here to view the file details")
    if file_button:
        st.write(file_details)
    st.markdown("--------------------")
    img = load_image(image_file)
    st.image(img, width=100)
    # Save the file in a temp location for the model to predict using
    img_name = 'temp'
    with open(os.path.join("tempDir", img_name), "wb") as f:
        f.write(image_file.getbuffer())
    st.success("Success..File successfully uploaded.")

# Calling the model to predict on the image
# Loading the model


def get_x(r): return PATH+r['image_name']
def get_y(r): return r['labels'].split(',')


def splitter(df):
    train = df.index[df['is_valid'] == 0].tolist()
    valid = df.index[df['is_valid'] == 1].tolist()
    return train, valid


class LabelSmoothingBCEWithLogitsLossFlat(BCEWithLogitsLossFlat):
    def __init__(self, eps: float = 0.1, **kwargs):
        self.eps = eps
        super().__init__(thresh=0.2, **kwargs)

    def __call__(self, inp, targ, **kwargs):
        targ_smooth = targ.float() * (1. - self.eps) + 0.5 * self.eps
        return super().__call__(inp, targ_smooth, **kwargs)

    def __repr__(self):
        return "FlattenedLoss of LabelSmoothingBCEWithLogits()"


# Trained model
model = load_learner("atr-recognition-stage-1-resnet34.pkl")


def predict_attribute(model, path):
    predicted = model.predict(path)
    return predicted[0]


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


if image_file is not None:
    PATH = "Tempdir/"
    image_path = PATH + img_name
    prediction = predict_attribute(model, image_path)
    # Since, OpenCV opens image in BGR mode, and we require RGB for correct image
    img = cv2.imread(image_path)
    # Also, we use cvtColor instead of imread
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    height, width, dim = img.shape
    #img = img[(height//3):(2*height//3), (width//3):(2*width//3), :]
    height, width, dim = img.shape
    img_vec = np.reshape(img, [height * width, dim])
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(img_vec)
    # Identify the colors
    unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
    # print(unique_l,counts_l)
    sort_ix = np.argsort(counts_l)
    sort_ix = sort_ix[::-1]
    for cluster_center in kmeans.cluster_centers_[sort_ix]:
        # print(cluster_center
        r = int(cluster_center[0])
        g = int(cluster_center[1])
        b = int(cluster_center[2])
        actual_color, closest_color = get_colour_name((r, g, b))
    color = closest_color
    st.write("Our model has predicted the dress to be of",
             prediction, "type and", color, "in color")

    # Recommended Dresses
    ind_cols1, ind_cols2, ind_cols3 = st.columns(3)
    with ind_cols2:
        st.header("Recommended Dresses")
    # Check with the common repository of dresses available
    # Since, I only hv one dataset at the moment. Ill try using that data alone.

    # Replace this with the common dataset for all companies
    data = pd.read_csv("JolieNoire-End.csv")

    selected_list_color = []
    brand_color = []
    product_title_color = []
    cost_color = []
    urls_color = []
    score_color = []

    for i in range(0, len(data)):
        score = 0
        color_check = data['color'][i]
        type_check = data['prediction'][i]
        for type in prediction:
            if type in type_check:
                score += 1
        if color in color_check:
            score += 1
            score_color.append(score)
            selected_list_color.append(data['Image Path'][i])
            brand_color.append(data['Company'][i])
            product_title_color.append(data['Product Name'][i])
            cost_color.append(data['Price'][i])
            urls_color.append(data['Product Link'][i])
    # Sort based on the score calculation for most similar first
    score_color, selected_list_color, brand_color, product_title_color, cost_color, urls_color = sort_together(
        [score_color, selected_list_color, brand_color, product_title_color, cost_color, urls_color], reverse=True)

    col1, col2, col3 = st.columns(3)
    for i in range(0, len(selected_list_color)):
        if i % 3 == 0:
            col1.image(
                selected_list_color[i], width=300, caption=str(brand_color[i] + ' || ' + product_title_color[i]))
            col1.caption(cost_color[i])
            if col1.button("Click here", key="key"+str(i)):
                webbrowser.open_new_tab(urls_color[i])
        if i % 3 == 1:
            col2.image(selected_list_color[i], width=300,
                       caption=str(brand_color[i] + ' || ' + product_title_color[i]))
            col2.caption(cost_color[i])
            if col2.button("Click here", key="key"+str(i)):
                webbrowser.open_new_tab(urls_color[i])
        if i % 3 == 2:
            col3.image(
                selected_list_color[i], width=300, caption=str(brand_color[i] + '|| ' + product_title_color[i]))
            col3.caption(cost_color[i])
            if col3.button("Click here", key="key"+str(i)):
                webbrowser.open_new_tab(urls_color[i])


else:
    st.write("Please upload an image to get related dresses")

st.markdown("--------------------")
