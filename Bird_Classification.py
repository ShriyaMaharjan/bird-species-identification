import os
import json
import librosa
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
from warnings import filterwarnings
model = load_model('C:\\Users\\HP\\Bird-Sound-Classification-using-Deep-Learning\\BC.h5', compile=False)

filterwarnings('ignore')
def streamlit_config():
    st.set_page_config(page_title="Bird Species Classification", layout="centered")


   
    page_background_color = """
    <style>
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

   
    st.markdown('<h1 style="text-align: center;">Bird Species Identification</h1>', unsafe_allow_html=True)
    add_vertical_space(2)


streamlit_config()


# Prediction functions
def prediction(audio_file, confidence_threshold=0.5):
    with open('prediction.json', mode='r') as f:
        prediction_dict = json.load(f)

    # Load and process the audio
    audio, sample_rate = librosa.load(audio_file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)
    mfccs_features = np.expand_dims(mfccs_features, axis=(0, 2))

    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    # Load the trained model
    model = tf.keras.models.load_model('model.h5')
    prediction = model.predict(mfccs_tensors)

    target_label = np.argmax(prediction)
    predicted_class = prediction_dict[str(target_label)]
    confidence = np.max(prediction)

    # If the confidence is below the threshold, we classify it as 'Not a Bird'
    if confidence < confidence_threshold:
        predicted_class = "Not a Bird"
        confidence = 0

    confidence = round(confidence * 100, 2)

    # Show results to the user
    add_vertical_space(1)
    st.markdown(f'<h4 style="text-align: center; color: orange;">{confidence}% Match Found</h4>', unsafe_allow_html=True)

    # If prediction is a bird, display the bird image, else show a message
    if predicted_class != "Not a Bird":
        image_path = os.path.join('Inference_Images', f'{predicted_class}.jpg')
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (350, 300))

        _, col2, _ = st.columns([0.1, 0.8, 0.1])
        with col2:
            st.image(img)

        st.markdown(f'<h3 style="text-align: center; color: green;">{predicted_class}</h3>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h3 style="text-align: center; color: red;">{predicted_class}</h3>', unsafe_allow_html=True)
        st.markdown('<h5 style="text-align: center;">No Bird Sound Detected. Please try again with a clearer bird sound.</h5>', unsafe_allow_html=True)

    return predicted_class

options = st.tabs(["Home", "About", "Identify a Bird"])

# import time

# # Splash screen logic
# placeholder = st.empty()  # Create a placeholder for the splash screen
# with placeholder.container():
#     st.markdown(
#         """
#         <div style='display: flex; justify-content: center; align-items: center; height: 50vh;'>
#             <h1 style='font-size: 50px;  text-align: center;'>Welcome to Bird Species Identification Project</h1>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
#     time.sleep(3)  # Display splash screen for 3 seconds

# # Remove the splash screen and display the actual content
# placeholder.empty()

# Home page content
with options[0]:
    st.image("bird.jpg", width=600, use_column_width=True)  # Full-width image
    st.markdown("<h1 style='text-align: center; color: white;'>Welcome to the Bird Species Identification Project!</h1>",
                unsafe_allow_html=True)

# About page content
with options[1]:
    st.markdown("<h2 style='text-align: center; '>About This Project</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.image("bird banner.jpg", use_column_width=True)
    st.markdown("""
        <p style='text-align: justify; font-size: 18px; color: #4d4d4d;'>
        The <strong>Bird Species Identification Project</strong> is designed to assist bird enthusiasts, researchers, and conservationists in identifying various bird species with ease. 
        Leveraging state-of-the-art machine learning techniques, this system enables accurate identification by analyzing both <strong>images</strong> and <strong>sounds</strong> of birds. 
        Whether you're an ornithologist studying bird populations or simply a hobbyist trying to recognize the birds in your backyard, this tool is your reliable companion.
        </p>
        <p style='text-align: justify; font-size: 18px; color: #4d4d4d;'>
        Built using advanced technologies like <strong>Convolutional Neural Networks (CNNs)</strong> for image classification and <strong>MFCC-based analysis</strong> for audio recognition, 
        the system combines the power of deep learning and digital signal processing to achieve high precision. With a user-friendly interface and intuitive design, users can easily upload bird photos or audio recordings to identify species from a comprehensive dataset.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; '>Key Features</h3>", unsafe_allow_html=True)
    st.markdown("""
        <ul style='font-size: 18px; color: #4d4d4d;'>
            <li>Support for both <strong>image</strong> and <strong>audio</strong> uploads for bird identification.</li>
            <li>State-of-the-art deep learning models trained on diverse bird datasets.</li>
            <li>High accuracy and fast predictions using optimized neural network architectures.</li>
            <li>Interactive and visually appealing interface built with <strong>Streamlit</strong>.</li>
            <li>Insightful results with confidence scores and species-specific details.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Benefits</h3>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: justify; font-size: 18px; '>
        By using this system, users can enjoy several benefits:
        </p>
        <ul style='font-size: 18px;'>
            <li>Identify bird species effortlessly, even for beginners.</li>
            <li>Contribute to bird conservation efforts by logging observations.</li>
            <li>Encourage environmental awareness and appreciation for biodiversity.</li>
            <li>Empower researchers with a reliable tool for data collection and analysis.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Future Enhancements</h3>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: justify; font-size: 18px; color: #4d4d4d;'>
        This project is a work in progress, and we aim to add more exciting features in the future, including:
        </p>
        <ul style='font-size: 18px; color: #4d4d4d;'>
            <li>Integration with live recording devices for real-time bird detection.</li>
            <li>Incorporation of geographic data to enhance species prediction accuracy based on location.</li>
            <li>Expanding the dataset to include rare and regional bird species.</li>
            <li>Adding multi-language support to cater to a global audience.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align: center; font-size: 18px; color: #4d4d4d;'>
        <strong>Join us in exploring the amazing world of birds!</strong>
        </p>
    """, unsafe_allow_html=True)

# Identify a Bird page content

with options[2]:
    st.markdown("<h2 style='text-align: center;'>Identify a Bird</h2>", unsafe_allow_html=True)

    
    identification_type = st.radio(
        "How would you like to identify the bird?",
        options=["Image", "Sound"],
        index=0
    )
    uploaded_audio=None
    uploaded_image=None

    
    st.write(f"Selected option: {identification_type}")
    if identification_type == "Sound":
        st.markdown("<h3 style='text-align: center; color: #2e8b57;'>", unsafe_allow_html=True)
        uploaded_audio = st.file_uploader("Upload a sound recording of the bird:", type=["wav", "mp3", "ogg"])
        
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        st.write("Sound uploaded. Processing sound identification...")

        
        identified_species = prediction(uploaded_audio)  
        st.write(f"Identified Bird Species: {identified_species}")  

    # if uploaded_audio is None:
    #     st.write("Please upload a sound file for identification.")
    # else:
    #         st.warning("Please upload a sound file for identification.")
    
    
lab = {0: 'ABBOTTS BABBLER',
 1: 'ABBOTTS BOOBY',
 2: 'ABYSSINIAN GROUND HORNBILL',
 3: 'AFRICAN CROWNED CRANE',
 4: 'AFRICAN EMERALD CUCKOO',
 5: 'AFRICAN FIREFINCH',
 6: 'AFRICAN OYSTER CATCHER',
 7: 'AFRICAN PIED HORNBILL',
 8: 'AFRICAN PYGMY GOOSE',
 9: 'ALBATROSS',
 10: 'ALBERTS TOWHEE',
 11: 'ALEXANDRINE PARAKEET',
 12: 'ALPINE CHOUGH',
 13: 'ALTAMIRA YELLOWTHROAT',
 14: 'AMERICAN AVOCET',
 15: 'AMERICAN BITTERN',
 16: 'AMERICAN COOT',
 17: 'AMERICAN DIPPER',
 18: 'AMERICAN FLAMINGO',
 19: 'AMERICAN GOLDFINCH',
 20: 'AMERICAN KESTREL',
 21: 'AMERICAN PIPIT',
 22: 'AMERICAN REDSTART',
 23: 'AMERICAN ROBIN',
 24: 'AMERICAN WIGEON',
 25: 'AMETHYST WOODSTAR',
 26: 'ANDEAN GOOSE',
 27: 'ANDEAN LAPWING',
 28: 'ANDEAN SISKIN',
 29: 'ANHINGA',
 30: 'ANIANIAU',
 31: 'ANNAS HUMMINGBIRD',
 32: 'ANTBIRD',
 33: 'ANTILLEAN EUPHONIA',
 34: 'APAPANE',
 35: 'APOSTLEBIRD',
 36: 'ARARIPE MANAKIN',
 37: 'ASHY STORM PETREL',
 38: 'ASHY THRUSHBIRD',
 39: 'ASIAN CRESTED IBIS',
 40: 'ASIAN DOLLARD BIRD',
 41: 'ASIAN GREEN BEE EATER',
 42: 'ASIAN OPENBILL STORK',
 43: 'AUCKLAND SHAQ',
 44: 'AUSTRAL CANASTERO',
 45: 'AUSTRALASIAN FIGBIRD',
 46: 'AVADAVAT',
 47: 'AZARAS SPINETAIL',
 48: 'AZURE BREASTED PITTA',
 49: 'AZURE JAY',
 50: 'AZURE TANAGER',
 51: 'AZURE TIT',
 52: 'BAIKAL TEAL',
 53: 'BALD EAGLE',
 54: 'BALD IBIS',
 55: 'BALI STARLING',
 56: 'BALTIMORE ORIOLE',
 57: 'BANANAQUIT',
 58: 'BAND TAILED GUAN',
 59: 'BANDED BROADBILL',
 60: 'BANDED PITA',
 61: 'BANDED STILT',
 62: 'BAR-TAILED GODWIT',
 63: 'BARN OWL',
 64: 'BARN SWALLOW',
 65: 'BARRED PUFFBIRD',
 66: 'BARROWS GOLDENEYE',
 67: 'BAY-BREASTED WARBLER',
 68: 'BEARDED BARBET',
 69: 'BEARDED BELLBIRD',
 70: 'BEARDED REEDLING',
 71: 'BELTED KINGFISHER',
 72: 'BIRD OF PARADISE',
 73: 'BLACK AND YELLOW BROADBILL',
 74: 'BLACK BAZA',
 75: 'BLACK BREASTED PUFFBIRD',
 76: 'BLACK COCKATO',
 77: 'BLACK FACED SPOONBILL',
 78: 'BLACK FRANCOLIN',
 79: 'BLACK HEADED CAIQUE',
 80: 'BLACK NECKED STILT',
 81: 'BLACK SKIMMER',
 82: 'BLACK SWAN',
 83: 'BLACK TAIL CRAKE',
 84: 'BLACK THROATED BUSHTIT',
 85: 'BLACK THROATED HUET',
 86: 'BLACK THROATED WARBLER',
 87: 'BLACK VENTED SHEARWATER',
 88: 'BLACK VULTURE',
 89: 'BLACK-CAPPED CHICKADEE',
 90: 'BLACK-NECKED GREBE',
 91: 'BLACK-THROATED SPARROW',
 92: 'BLACKBURNIAM WARBLER',
 93: 'BLONDE CRESTED WOODPECKER',
 94: 'BLOOD PHEASANT',
 95: 'BLUE COAU',
 96: 'BLUE DACNIS',
 97: 'BLUE GRAY GNATCATCHER',
 98: 'BLUE GROSBEAK',
 99: 'BLUE GROUSE',
 100: 'BLUE HERON',
 101: 'BLUE MALKOHA',
 102: 'BLUE THROATED PIPING GUAN',
 103: 'BLUE THROATED TOUCANET',
 104: 'BOBOLINK',
 105: 'BORNEAN BRISTLEHEAD',
 106: 'BORNEAN LEAFBIRD',
 107: 'BORNEAN PHEASANT',
 108: 'BRANDT CORMARANT',
 109: 'BREWERS BLACKBIRD',
 110: 'BROWN CREPPER',
 111: 'BROWN HEADED COWBIRD',
 112: 'BROWN NOODY',
 113: 'BROWN THRASHER',
 114: 'BUFFLEHEAD',
 115: 'BULWERS PHEASANT',
 116: 'BURCHELLS COURSER',
 117: 'BUSH TURKEY',
 118: 'CAATINGA CACHOLOTE',
 119: 'CABOTS TRAGOPAN',
 120: 'CACTUS WREN',
 121: 'CALIFORNIA CONDOR',
 122: 'CALIFORNIA GULL',
 123: 'CALIFORNIA QUAIL',
 124: 'CAMPO FLICKER',
 125: 'CANARY',
 126: 'CANVASBACK',
 127: 'CAPE GLOSSY STARLING',
 128: 'CAPE LONGCLAW',
 129: 'CAPE MAY WARBLER',
 130: 'CAPE ROCK THRUSH',
 131: 'CAPPED HERON',
 132: 'CAPUCHINBIRD',
 133: 'CARMINE BEE-EATER',
 134: 'CASPIAN TERN',
 135: 'CASSOWARY',
 136: 'CEDAR WAXWING',
 137: 'CERULEAN WARBLER',
 138: 'CHARA DE COLLAR',
 139: 'CHATTERING LORY',
 140: 'CHESTNET BELLIED EUPHONIA',
 141: 'CHESTNUT WINGED CUCKOO',
 142: 'CHINESE BAMBOO PARTRIDGE',
 143: 'CHINESE POND HERON',
 144: 'CHIPPING SPARROW',
 145: 'CHUCAO TAPACULO',
 146: 'CHUKAR PARTRIDGE',
 147: 'CINNAMON ATTILA',
 148: 'CINNAMON FLYCATCHER',
 149: 'CINNAMON TEAL',
 150: 'CLARKS GREBE',
 151: 'CLARKS NUTCRACKER',
 152: 'COCK OF THE  ROCK',
 153: 'COCKATOO',
 154: 'COLLARED ARACARI',
 155: 'COLLARED CRESCENTCHEST',
 156: 'COMMON FIRECREST',
 157: 'COMMON GRACKLE',
 158: 'COMMON HOUSE MARTIN',
 159: 'COMMON IORA',
 160: 'COMMON LOON',
 161: 'COMMON POORWILL',
 162: 'COMMON STARLING',
 163: 'COPPERSMITH BARBET',
 164: 'COPPERY TAILED COUCAL',
 165: 'CRAB PLOVER',
 166: 'CRANE HAWK',
 167: 'CREAM COLORED WOODPECKER',
 168: 'CRESTED AUKLET',
 169: 'CRESTED CARACARA',
 170: 'CRESTED COUA',
 171: 'CRESTED FIREBACK',
 172: 'CRESTED KINGFISHER',
 173: 'CRESTED NUTHATCH',
 174: 'CRESTED OROPENDOLA',
 175: 'CRESTED SERPENT EAGLE',
 176: 'CRESTED SHRIKETIT',
 177: 'CRESTED WOOD PARTRIDGE',
 178: 'CRIMSON CHAT',
 179: 'CRIMSON SUNBIRD',
 180: 'CROW',
 181: 'CUBAN TODY',
 182: 'CUBAN TROGON',
 183: 'CURL CRESTED ARACURI',
 184: 'D-ARNAUDS BARBET',
 185: 'DALMATIAN PELICAN',
 186: 'DARJEELING WOODPECKER',
 187: 'DARK EYED JUNCO',
 188: 'DAURIAN REDSTART',
 189: 'DEMOISELLE CRANE',
 190: 'DOUBLE BARRED FINCH',
 191: 'DOUBLE BRESTED CORMARANT',
 192: 'DOUBLE EYED FIG PARROT',
 193: 'DOWNY WOODPECKER',
 194: 'DUNLIN',
 195: 'DUSKY LORY',
 196: 'DUSKY ROBIN',
 197: 'EARED PITA',
 198: 'EASTERN BLUEBIRD',
 199: 'EASTERN BLUEBONNET',
 200: 'EASTERN GOLDEN WEAVER',
 201: 'EASTERN MEADOWLARK',
 202: 'EASTERN ROSELLA',
 203: 'EASTERN TOWEE',
 204: 'EASTERN WIP POOR WILL',
 205: 'EASTERN YELLOW ROBIN',
 206: 'ECUADORIAN HILLSTAR',
 207: 'EGYPTIAN GOOSE',
 208: 'ELEGANT TROGON',
 209: 'ELLIOTS  PHEASANT',
 210: 'EMERALD TANAGER',
 211: 'EMPEROR PENGUIN',
 212: 'EMU',
 213: 'ENGGANO MYNA',
 214: 'EURASIAN BULLFINCH',
 215: 'EURASIAN GOLDEN ORIOLE',
 216: 'EURASIAN MAGPIE',
 217: 'EUROPEAN GOLDFINCH',
 218: 'EUROPEAN TURTLE DOVE',
 219: 'EVENING GROSBEAK',
 220: 'FAIRY BLUEBIRD',
 221: 'FAIRY PENGUIN',
 222: 'FAIRY TERN',
 223: 'FAN TAILED WIDOW',
 224: 'FASCIATED WREN',
 225: 'FIERY MINIVET',
 226: 'FIORDLAND PENGUIN',
 227: 'FIRE TAILLED MYZORNIS',
 228: 'FLAME BOWERBIRD',
 229: 'FLAME TANAGER',
 230: 'FOREST WAGTAIL',
 231: 'FRIGATE',
 232: 'FRILL BACK PIGEON',
 233: 'GAMBELS QUAIL',
 234: 'GANG GANG COCKATOO',
 235: 'GILA WOODPECKER',
 236: 'GILDED FLICKER',
 237: 'GLOSSY IBIS',
 238: 'GO AWAY BIRD',
 239: 'GOLD WING WARBLER',
 240: 'GOLDEN BOWER BIRD',
 241: 'GOLDEN CHEEKED WARBLER',
 242: 'GOLDEN CHLOROPHONIA',
 243: 'GOLDEN EAGLE',
 244: 'GOLDEN PARAKEET',
 245: 'GOLDEN PHEASANT',
 246: 'GOLDEN PIPIT',
 247: 'GOULDIAN FINCH',
 248: 'GRANDALA',
 249: 'GRAY CATBIRD',
 250: 'GRAY KINGBIRD',
 251: 'GRAY PARTRIDGE',
 252: 'GREAT ARGUS',
 253: 'GREAT GRAY OWL',
 254: 'GREAT JACAMAR',
 255: 'GREAT KISKADEE',
 256: 'GREAT POTOO',
 257: 'GREAT TINAMOU',
 258: 'GREAT XENOPS',
 259: 'GREATER PEWEE',
 260: 'GREATER PRAIRIE CHICKEN',
 261: 'GREATOR SAGE GROUSE',
 262: 'GREEN BROADBILL',
 263: 'GREEN JAY',
 264: 'GREEN MAGPIE',
 265: 'GREEN WINGED DOVE',
 266: 'GREY CUCKOOSHRIKE',
 267: 'GREY HEADED CHACHALACA',
 268: 'GREY HEADED FISH EAGLE',
 269: 'GREY PLOVER',
 270: 'GROVED BILLED ANI',
 271: 'GUINEA TURACO',
 272: 'GUINEAFOWL',
 273: 'GURNEYS PITTA',
 274: 'GYRFALCON',
 275: 'HAMERKOP',
 276: 'HARLEQUIN DUCK',
 277: 'HARLEQUIN QUAIL',
 278: 'HARPY EAGLE',
 279: 'HAWAIIAN GOOSE',
 280: 'HAWFINCH',
 281: 'HELMET VANGA',
 282: 'HEPATIC TANAGER',
 283: 'HIMALAYAN BLUETAIL',
 284: 'HIMALAYAN MONAL',
 285: 'HOATZIN',
 286: 'HOODED MERGANSER',
 287: 'HOOPOES',
 288: 'HORNED GUAN',
 289: 'HORNED LARK',
 290: 'HORNED SUNGEM',
 291: 'HOUSE FINCH',
 292: 'HOUSE SPARROW',
 293: 'HYACINTH MACAW',
 294: 'IBERIAN MAGPIE',
 295: 'IBISBILL',
 296: 'IMPERIAL SHAQ',
 297: 'INCA TERN',
 298: 'INDIAN BUSTARD',
 299: 'INDIAN PITTA',
 300: 'INDIAN ROLLER',
 301: 'INDIAN VULTURE',
 302: 'INDIGO BUNTING',
 303: 'INDIGO FLYCATCHER',
 304: 'INLAND DOTTEREL',
 305: 'IVORY BILLED ARACARI',
 306: 'IVORY GULL',
 307: 'IWI',
 308: 'JABIRU',
 309: 'JACK SNIPE',
 310: 'JACOBIN PIGEON',
 311: 'JANDAYA PARAKEET',
 312: 'JAPANESE ROBIN',
 313: 'JAVA SPARROW',
 314: 'JOCOTOCO ANTPITTA',
 315: 'KAGU',
 316: 'KAKAPO',
 317: 'KILLDEAR',
 318: 'KING EIDER',
 319: 'KING VULTURE',
 320: 'KIWI',
 321: 'KNOB BILLED DUCK',
 322: 'KOOKABURRA',
 323: 'LARK BUNTING',
 324: 'LAUGHING GULL',
 325: 'LAZULI BUNTING',
 326: 'LESSER ADJUTANT',
 327: 'LILAC ROLLER',
 328: 'LIMPKIN',
 329: 'LITTLE AUK',
 330: 'LOGGERHEAD SHRIKE',
 331: 'LONG-EARED OWL',
 332: 'LOONEY BIRDS',
 333: 'LUCIFER HUMMINGBIRD',
 334: 'MAGPIE GOOSE',
 335: 'MALABAR HORNBILL',
 336: 'MALACHITE KINGFISHER',
 337: 'MALAGASY WHITE EYE',
 338: 'MALEO',
 339: 'MALLARD DUCK',
 340: 'MANDRIN DUCK',
 341: 'MANGROVE CUCKOO',
 342: 'MARABOU STORK',
 343: 'MASKED BOBWHITE',
 344: 'MASKED BOOBY',
 345: 'MASKED LAPWING',
 346: 'MCKAYS BUNTING',
 347: 'MERLIN',
 348: 'MIKADO  PHEASANT',
 349: 'MILITARY MACAW',
 350: 'MOURNING DOVE',
 351: 'MYNA',
 352: 'NICOBAR PIGEON',
 353: 'NOISY FRIARBIRD',
 354: 'NORTHERN BEARDLESS TYRANNULET',
 355: 'NORTHERN CARDINAL',
 356: 'NORTHERN FLICKER',
 357: 'NORTHERN FULMAR',
 358: 'NORTHERN GANNET',
 359: 'NORTHERN GOSHAWK',
 360: 'NORTHERN JACANA',
 361: 'NORTHERN MOCKINGBIRD',
 362: 'NORTHERN PARULA',
 363: 'NORTHERN RED BISHOP',
 364: 'NORTHERN SHOVELER',
 365: 'OCELLATED TURKEY',
 366: 'OILBIRD',
 367: 'OKINAWA RAIL',
 368: 'ORANGE BREASTED TROGON',
 369: 'ORANGE BRESTED BUNTING',
 370: 'ORIENTAL BAY OWL',
 371: 'ORNATE HAWK EAGLE',
 372: 'OSPREY',
 373: 'OSTRICH',
 374: 'OVENBIRD',
 375: 'OYSTER CATCHER',
 376: 'PAINTED BUNTING',
 377: 'PALILA',
 378: 'PALM NUT VULTURE',
 379: 'PARADISE TANAGER',
 380: 'PARAKETT  AUKLET',
 381: 'PARUS MAJOR',
 382: 'PATAGONIAN SIERRA FINCH',
 383: 'PEACOCK',
 384: 'PEREGRINE FALCON',
 385: 'PHAINOPEPLA',
 386: 'PHILIPPINE EAGLE',
 387: 'PINK ROBIN',
 388: 'PLUSH CRESTED JAY',
 389: 'POMARINE JAEGER',
 390: 'PUFFIN',
 391: 'PUNA TEAL',
 392: 'PURPLE FINCH',
 393: 'PURPLE GALLINULE',
 394: 'PURPLE MARTIN',
 395: 'PURPLE SWAMPHEN',
 396: 'PYGMY KINGFISHER',
 397: 'PYRRHULOXIA',
 398: 'QUETZAL',
 399: 'RAINBOW LORIKEET',
 400: 'RAZORBILL',
 401: 'RED BEARDED BEE EATER',
 402: 'RED BELLIED PITTA',
 403: 'RED BILLED TROPICBIRD',
 404: 'RED BROWED FINCH',
 405: 'RED CROSSBILL',
 406: 'RED FACED CORMORANT',
 407: 'RED FACED WARBLER',
 408: 'RED FODY',
 409: 'RED HEADED DUCK',
 410: 'RED HEADED WOODPECKER',
 411: 'RED KNOT',
 412: 'RED LEGGED HONEYCREEPER',
 413: 'RED NAPED TROGON',
 414: 'RED SHOULDERED HAWK',
 415: 'RED TAILED HAWK',
 416: 'RED TAILED THRUSH',
 417: 'RED WINGED BLACKBIRD',
 418: 'RED WISKERED BULBUL',
 419: 'REGENT BOWERBIRD',
 420: 'RING-NECKED PHEASANT',
 421: 'ROADRUNNER',
 422: 'ROCK DOVE',
 423: 'ROSE BREASTED COCKATOO',
 424: 'ROSE BREASTED GROSBEAK',
 425: 'ROSEATE SPOONBILL',
 426: 'ROSY FACED LOVEBIRD',
 427: 'ROUGH LEG BUZZARD',
 428: 'ROYAL FLYCATCHER',
 429: 'RUBY CROWNED KINGLET',
 430: 'RUBY THROATED HUMMINGBIRD',
 431: 'RUDDY SHELDUCK',
 432: 'RUDY KINGFISHER',
 433: 'RUFOUS KINGFISHER',
 434: 'RUFOUS TREPE',
 435: 'RUFUOS MOTMOT',
 436: 'SAMATRAN THRUSH',
 437: 'SAND MARTIN',
 438: 'SANDHILL CRANE',
 439: 'SATYR TRAGOPAN',
 440: 'SAYS PHOEBE',
 441: 'SCARLET CROWNED FRUIT DOVE',
 442: 'SCARLET FACED LIOCICHLA',
 443: 'SCARLET IBIS',
 444: 'SCARLET MACAW',
 445: 'SCARLET TANAGER',
 446: 'SHOEBILL',
 447: 'SHORT BILLED DOWITCHER',
 448: 'SMITHS LONGSPUR',
 449: 'SNOW GOOSE',
 450: 'SNOW PARTRIDGE',
 451: 'SNOWY EGRET',
 452: 'SNOWY OWL',
 453: 'SNOWY PLOVER',
 454: 'SNOWY SHEATHBILL',
 455: 'SORA',
 456: 'SPANGLED COTINGA',
 457: 'SPLENDID WREN',
 458: 'SPOON BILED SANDPIPER',
 459: 'SPOTTED CATBIRD',
 460: 'SPOTTED WHISTLING DUCK',
 461: 'SQUACCO HERON',
 462: 'SRI LANKA BLUE MAGPIE',
 463: 'STEAMER DUCK',
 464: 'STORK BILLED KINGFISHER',
 465: 'STRIATED CARACARA',
 466: 'STRIPED OWL',
 467: 'STRIPPED MANAKIN',
 468: 'STRIPPED SWALLOW',
 469: 'SUNBITTERN',
 470: 'SUPERB STARLING',
 471: 'SURF SCOTER',
 472: 'SWINHOES PHEASANT',
 473: 'TAILORBIRD',
 474: 'TAIWAN MAGPIE',
 475: 'TAKAHE',
 476: 'TASMANIAN HEN',
 477: 'TAWNY FROGMOUTH',
 478: 'TEAL DUCK',
 479: 'TIT MOUSE',
 480: 'TOUCHAN',
 481: 'TOWNSENDS WARBLER',
 482: 'TREE SWALLOW',
 483: 'TRICOLORED BLACKBIRD',
 484: 'TROPICAL KINGBIRD',
 485: 'TRUMPTER SWAN',
 486: 'TURKEY VULTURE',
 487: 'TURQUOISE MOTMOT',
 488: 'UMBRELLA BIRD',
 489: 'VARIED THRUSH',
 490: 'VEERY',
 491: 'VENEZUELIAN TROUPIAL',
 492: 'VERDIN',
 493: 'VERMILION FLYCATHER',
 494: 'VICTORIA CROWNED PIGEON',
 495: 'VIOLET BACKED STARLING',
 496: 'VIOLET CUCKOO',
 497: 'VIOLET GREEN SWALLOW',
 498: 'VIOLET TURACO',
 499: 'VISAYAN HORNBILL',
 500: 'VULTURINE GUINEAFOWL',
 501: 'WALL CREAPER',
 502: 'WATTLED CURASSOW',
 503: 'WATTLED LAPWING',
 504: 'WHIMBREL',
 505: 'WHITE BREASTED WATERHEN',
 506: 'WHITE BROWED CRAKE',
 507: 'WHITE CHEEKED TURACO',
 508: 'WHITE CRESTED HORNBILL',
 509: 'WHITE EARED HUMMINGBIRD',
 510: 'WHITE NECKED RAVEN',
 511: 'WHITE TAILED TROPIC',
 512: 'WHITE THROATED BEE EATER',
 513: 'WILD TURKEY',
 514: 'WILLOW PTARMIGAN',
 515: 'WILSONS BIRD OF PARADISE',
 516: 'WOOD DUCK',
 517: 'WOOD THRUSH',
 518: 'WOODLAND KINGFISHER',
 519: 'WRENTIT',
 520: 'YELLOW BELLIED FLOWERPECKER',
 521: 'YELLOW BREASTED CHAT',
 522: 'YELLOW CACIQUE',
 523: 'YELLOW HEADED BLACKBIRD',
 524: 'ZEBRA DOVE'}



model = load_model("BC.h5")  
confidence_threshold = 0.5  
descriptions_path = 'C:\\Users\\HP\\Bird-species-detection\\Birds_description\\{}.txt'  # Path to descriptions


def preprocess_image(image):
    image = image.resize((224, 224))  
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  
    image_array = image_array / 255.0  
    return image_array


def predict_and_describe(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    if confidence < confidence_threshold:
        return "Unknown species", "The confidence level is too low to identify the species."
    
    bird_species = lab.get(predicted_class, "Unknown species")
    
    # Load the bird's description
    description_file = descriptions_path.format(bird_species)
    if os.path.exists(description_file):
        with open(description_file, 'r') as file:
            description = file.read()
    else:
        description = "Description not available for this species."
    
    return bird_species, description


def run():

    # img1 = Image.open('C:\\Users\\HP\\Bird-species-detection\\meta\\logo1.png')
    # img1 = img1.resize((350, 350))
    # st.image(img1, use_column_width=False)
    # st.title("Bird Species Classification")
    # st.markdown(
    #     '''<h4 style='text-align: left; color: #d73b5c;'>* Dataset is based on 525 Bird Species</h4>''',
    #     unsafe_allow_html=True
    # )

    # Image upload
    uploaded_file = st.file_uploader("Upload a bird image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict"):
            try:
                bird_species, description = predict_and_describe(image)
                st.success(f"Predicted Bird Species: {bird_species}")
                st.markdown("### Description:")
                st.write(description)
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Execute the app
if __name__ == "__main__":
    run()