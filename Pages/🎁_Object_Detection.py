import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
import imutils
import streamlit as st
st.set_page_config(
    page_title="Obeject Detedtion Using YOLOv7",
    page_icon="🔤",
    layout = "wide",
    initial_sidebar_state="expanded")
random.seed(123)

#######################################################################################################################################
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def pre_process(img):# pre processing the image given to the data set 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # since the image is read in bgr by opencv , converting it to regular RGB
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    # scaling our data . since the pixels are out of 255 dividing it by 255 give a number betwwen 0 and 1
    im /= 255 
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]:im} # im is the numpy array of input image pixels and inname is the name of the input image
    return outname, inp , image, ratio, dwdh 

def post_process(img ,outputs , dwdh , ratio ,conf_thres , nms_thres , nms_need):# post processing the data 
    ori_images = [img.copy()]#original image
    mask_img = [img.copy()]# image that is goona be used of masking purpose later on 
    OUT_TEXT.text(f"Number os Object Detected :- {len(outputs)}")
    if not nms_need:
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):#outputs contains the results from the yolo model 
            if score>conf_thres :
                image = ori_images[int(batch_id)]
                mask_image = mask_img[int(batch_id)]

                box = np.array([x0,y0,x1,y1])# getting the 4 co-ord for making the detection box
                box -= np.array(dwdh*2)
                box /= ratio # type: ignore    
                box = box.round().astype(np.int32).tolist()

                cls_id = int(cls_id)
                score = round(float(score),3)
                try:
                    name = names[cls_id]
                    color = colors[name]
                    name += ' '+str(score)

                    cv2.rectangle(image,box[:2],box[2:],color,1, lineType=cv2.LINE_AA)#creating boxes around the detected obejects with nobg
                    cv2.rectangle(mask_image,box[:2],box[2:],color,-1, lineType=cv2.LINE_AA)#creating boxes around the detected obejects with bg

                    #creating text above the detected obejects with a bg
                    t_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, thickness=1)[0]#used to get estimate abt how much size the text is going to take 
                    text_w, text_h = t_size
                    x_,y_=box[:2]
                    cv2.rectangle(image,box[:2],(x_+text_w+2 , y_- text_h -3),color,-1)
                    cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.4,[225, 255, 255],thickness=0) 
                    
                    x,y,w,h = box
                    cut_out_img = img[y:h, x:w]
                    height, width, channels = cut_out_img.shape
                    if height!= 0 and width != 0  and channels != 0 :
                        cut_out_img = cv2.resize(cut_out_img, None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
                        dis2.image(cut_out_img)
                        col2.text(name)
                except :
                    print("Not present in given names")
                    # cv2.imshow("plate", cut_out_img)
                
                
                
    else:
        indices = cv2.dnn.NMSBoxes(outputs[:,1:5], outputs[:,-1], conf_thres, nms_thres).flatten()# getting non maximum suppression

        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs[indices]):
            if score>conf_thres :
                image = ori_images[int(batch_id)]
                mask_image = mask_img[int(batch_id)]

                box = np.array([x0,y0,x1,y1])# getting the 4 co-ord for making the detection box
                box -= np.array(dwdh*2)
                box /= ratio # type: ignore    
                box = box.round().astype(np.int32).tolist()

                cls_id = int(cls_id)
                score = round(float(score),3)
                name = names[cls_id]
                color = colors[name]
                name += ' '+str(score)

                cv2.rectangle(image,box[:2],box[2:],color,1, lineType=cv2.LINE_AA)#creating boxes around the detected obejects with nobg
                cv2.rectangle(mask_image,box[:2],box[2:],color,-1, lineType=cv2.LINE_AA)#creating boxes around the detected obejects with bg

                #creating text above the detected obejects with a bg
                t_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, thickness=1)[0]#used to get estimate abt how much size the text is going to take 
                text_w, text_h = t_size
                x_,y_=box[:2]
                cv2.rectangle(image,box[:2],(x_+text_w+2 , y_- text_h -3),color,-1)
                cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.4,[225, 255, 255],thickness=0) 

    return ori_images[0],mask_img[0]

########################################################################################################################################################################

def start_process(Cuda , Weights , Source , Conf_thres , Nms_thres , Need_of_NMS , Need_of_mask , Cv2_wanted , File_Type , Names):
    global names , colors , session
    cuda = Cuda
    w = Weights
    cv2_want = Cv2_wanted
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    
    if providers == ['CUDAExecutionProvider', 'CPUExecutionProvider']:
        print("Running on GPU !!!!")
    else:print("Running in CPU")
    
    session = ort.InferenceSession(w, providers=providers)
    names = Names
    # names present in the coco.names dataset
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

    #genrating random colors for each item in the data set
    
    conf_thres= Conf_thres# Denotes the amout of confidence needed for the obeject to detected to be showed
    nms_thres= Nms_thres
    nms_need = Need_of_NMS # Used for Non Maximum Suppression so that the boxes dont overlap each other
    want_mask_img = Need_of_mask
    file_type = File_Type

    if file_type == "Video":
        pre_fps = 0
        fps= 0 
        new_fps = 0

        cap = cv2.VideoCapture(Source)
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
          print("Error opening video stream or file")

        # Read until video is completed
        while(cap.isOpened()):
        # Capture frame-by-frame
        # Press key q to stop
            new_fps = time.time()
            fps = round(1/(new_fps-pre_fps),2)
            print("fps :" ,fps)
            FPS.text(f"FPS :- {fps}")
            #fps=str(fps)
            pre_fps = new_fps
                        
            if cv2.waitKey(1) == ord('q'): #press q in the keyboard to exit.
                break
            try:
                # Read frame from the video
                ret, frame = cap.read()
                img = frame
                if not ret:
                    break
            except Exception as e:
                print(e)
                continue
            outname, inp , image, ratio, dwdh  = pre_process(img)
            # ONNX inference -- calling and using the converted Yolo - Onnx model 

            outputs = session.run(outname, inp)[0]
            ori_images,mask_img = post_process(img , outputs , dwdh , ratio,conf_thres ,nms_thres , nms_need)
            masked_img = cv2.addWeighted(ori_images,0.8,mask_img,0.2,1)
            masked_img = cv2.putText(masked_img,"fps : " + str(fps) ,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,[225, 255, 255],thickness=0)
            ori_images = cv2.putText(ori_images,"fps : " + str(fps) ,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,[225, 255, 255],thickness=0)
            
            if cv2_want:
                if want_mask_img:
                    cv2.imshow("Output : Masked Version",masked_img)
                else:
                    cv2.imshow("Output : Image",ori_images)
            else:
                if want_mask_img:
                    Output_Img  = masked_img
                else:
                    Output_Img = ori_images
                dis1.image(cv2.cvtColor(Output_Img, cv2.COLOR_BGR2RGB))

        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
    elif file_type == "Image":
        t0 = time.time()
        img = cv2.imread(Source)
        height, width, channels = img.shape
        if width > 600 :
            img = imutils.resize(img, width=600)
        
        outname, inp , image, ratio, dwdh  = pre_process(img)
        # ONNX inference -- calling and using the converted Yolo - Onnx model 

        outputs = session.run(outname, inp)[0]
        ori_images,mask_img = post_process(img , outputs , dwdh , ratio,conf_thres ,nms_thres , nms_need)

        masked_img = cv2.addWeighted(ori_images,0.8,mask_img,0.2,1)
        #cv2.imshow("image",cv2.cvtColor(ori_images[0], cv2.COLOR_BGR2RGB) )
        print(f"time Taken : {round(time.time()-t0 , 2 )}")
        if cv2_want:
            if want_mask_img:
                    cv2.imshow("Output : Masked Version",masked_img)
            else:
                    cv2.imshow("Output : Image",ori_images)
        else:
            if want_mask_img:
                    Output_Img  = masked_img
            else:
                    Output_Img = ori_images
            dis1.image(cv2.cvtColor(Output_Img, cv2.COLOR_BGR2RGB))
    else:
        print("Somthing is wrong with File Type")
####################################################################################################################################

if __name__ == "__main__":  
    st.markdown("""
            <style>
                .sidebar .sidebar-content {
                    width: 375px;
                }
                .big-font {
                    font-size:80px;
                    font-weight : 1000;
                }
                .small-font {
                    font-size:40px;
                    font-weight : 700;
                }
                .MuiGrid-item{
                    font-size:19px;
                }
                .css-1yy6isu p{
                    font-size:25px;
                }
                .st-dx{
                    font-size :18px;
                }
                .css-1fv8s86 p{
                    font-size:18px;
                }
            </style>""", unsafe_allow_html=True)

    st.markdown('<p class="big-font"><center class="big-font">Object Detection Using YOLOv7</center></p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font"><center class="small-font">Made by :- Srivatsa Gorti</center></p>', unsafe_allow_html=True)
    st.markdown("""---""")
    # st.markdown("**:green[My GitHub]** 📖: https://github.com/srivatsacool ")
    #st.markdown(":green[$\sqrt{x^2+y^2}=1$] is a Pythagorean identity. :pencil:")

    image = Image.open('Pages/dis1_display.png')
    image2 = Image.open('Pages/dis2_display.jpg')
    cv2_want = False
    with st.sidebar:
        CUDA = st.checkbox("CUDA ( Use GPU )",value = True)
        NMS = st.checkbox("NMS ( Non Maximum Suppression )",value = False)
        MASK = st.checkbox("Create Mask",value = True)
        OCR = st.checkbox("OCR",value = True)
        CONF_THRES = st.slider(
            'Confidence Threshold of the Model',0.0, 1.0, 0.5)
        st.write('Conf. Threshold :', CONF_THRES)
        NMS_THRES = st.slider(
            'NMS Threshold of the Model',0.0, 1.0, 0.5)
        st.write('NMS Threshold :', NMS_THRES) 
        WEIGHTS = st.selectbox('Choose the weights for the Model',
        ('Pages/yolov7.onnx','Pages/yolov7-e6e.onnx'),index = 0)
        
        li =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']
        NAMES = st.multiselect("What object do you want to detect" , ["All",'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush'] , ["All"])
        if 'All' in NAMES:
            NAMES = li
        st.text(NAMES)
        st.markdown("""---""")

        FILE_TYPE = st.radio("Detection On :-",('Image', 'Video'))
        st.markdown("""---""")

        UPLOAD = st.file_uploader("Choose a file (Supports only .mp4 , .jpg )" , type = ['jpg','mp4'])
        EXAMPLES = st.selectbox('Would like to use our Examples !!',
        ('Pages/People.mp4','Pages/car2.mp4','Pages/traffic.jpg','Pages/NewYork TimesSquare.jpg'),index = 2)
        if UPLOAD != None:
            if FILE_TYPE=='Image':
                st.image(UPLOAD)
                SOURCE = UPLOAD.name
                st.text(SOURCE)
            elif FILE_TYPE =='Video':
                SOURCE = UPLOAD.name
                st.text(UPLOAD)
                st.video(open(SOURCE,'rb').read())
        else:
            SOURCE = EXAMPLES
            if FILE_TYPE=='Image':
                st.image(SOURCE)
            elif FILE_TYPE =='Video':
                st.video(open(SOURCE,'rb').read())
            st.text(SOURCE)
    # if FILE_TYPE=='Image':


    col1, col2 = st.columns([3, 1])

    if FILE_TYPE=='Image':
        dis1 = col1.image(SOURCE)
    elif FILE_TYPE =='Video':
        dis1 = col1.video(open(SOURCE,'rb').read())
        
    col1.markdown("""<center>Output file</center>""", unsafe_allow_html=True)
    if FILE_TYPE =="Video":
        FPS = st.text("To see the FPS , press 'Start' !!")
    # col2.markdown("""<br>""", unsafe_allow_html=True)
    dis2 = col2.image(image2 , caption = "Detected Object")
    col2.markdown("""---""")
    col2.subheader("Obejects Detected with Conf level :-")
    OUT_TEXT = col2.text("Start to see the details...")

    st.markdown("----", unsafe_allow_html=True)
    columns = st.columns((2, 1, 2))
    button_pressed = columns[1].button('Start !!')
    st.markdown("----", unsafe_allow_html=True)
    if button_pressed:
        columns[1].text("Starting")
        start_process(CUDA , WEIGHTS , SOURCE , CONF_THRES , NMS_THRES , NMS , MASK ,cv2_want, FILE_TYPE  ,NAMES )
        
    st.subheader("Breif Description :")
    st.markdown("""
                - Based on state-of-art object detection model **:blue['YOLOv7']**, latest in the YOLO family .
                - Trained on a Custom Dataset that I have annotated myself , consists of 1000 train , 1000 val , 500 test images .
                - More details in **:green[GitHub]** Repo :- .
                """)