
import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageTk

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import mysql.connector
from datetime import datetime




import serial

ser = serial.Serial(port='COM4', baudrate=9600)  # timeout=.1

image = cv2.imread("beyaz.png")

root = tk.Tk()

# setting title
root.title("undefined")
# setting window size
width = 1659
height = 563
screenwidth = root.winfo_screenwidth()
screenheight = root.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
root.geometry(alignstr)
root.resizable(width=False, height=False)

GLabel_854 = tk.Label(root)
GLabel_854["bg"] = "#d52525"
ft = tkFont.Font(family='Times', size=10)
GLabel_854["font"] = ft
GLabel_854["fg"] = "#333333"
GLabel_854["justify"] = "center"
GLabel_854["text"] = ""
GLabel_854.place(x=20, y=20, width=637, height=477)

GLabel_224 = tk.Label(root)
ft = tkFont.Font(family='Times', size=11)
GLabel_224["font"] = ft
GLabel_224["fg"] = "#333333"
GLabel_224["justify"] = "center"
GLabel_224["text"] = "Ürün Bulunamadı"
GLabel_224["relief"] = "groove"
GLabel_224.place(x=740, y=90, width=200, height=40)

GLabel_282 = tk.Label(root)
ft = tkFont.Font(family='Times', size=11)
GLabel_282["font"] = ft
GLabel_282["fg"] = "#333333"
GLabel_282["justify"] = "center"
GLabel_282["text"] = "0"
GLabel_282["relief"] = "groove"
GLabel_282.place(x=740, y=200, width=200, height=40)

GLabel_624 = tk.Label(root)
ft = tkFont.Font(family='Times', size=16)
GLabel_624["font"] = ft
GLabel_624["fg"] = "#333333"
GLabel_624["justify"] = "center"
GLabel_624["text"] = "Ürün"
GLabel_624.place(x=740, y=40, width=200, height=40)

GLabel_66 = tk.Label(root)
ft = tkFont.Font(family='Times', size=16)
GLabel_66["font"] = ft
GLabel_66["fg"] = "#333333"
GLabel_66["justify"] = "center"
GLabel_66["text"] = "Ürün Ağırlığı"
GLabel_66.place(x=740, y=150, width=200, height=50)

GLabel_823 = tk.Label(root)
ft = tkFont.Font(family='Times', size=16)
GLabel_823["font"] = ft
GLabel_823["fg"] = "#333333"
GLabel_823["justify"] = "center"
GLabel_823["text"] = "Anlık Değer"
GLabel_823.place(x=740, y=250, width=200, height=50)

GLabel_929 = tk.Label(root)
ft = tkFont.Font(family='Times', size=11)
GLabel_929["font"] = ft
GLabel_929["fg"] = "#333333"
GLabel_929["justify"] = "center"
GLabel_929["text"] = "0"
GLabel_929["relief"] = "groove"
GLabel_929.place(x=740, y=300, width=200, height=40)

GLabel_434 = tk.Label(root)
ft = tkFont.Font(family='Times', size=16)
GLabel_434["font"] = ft
GLabel_434["fg"] = "#333333"
GLabel_434["justify"] = "center"
GLabel_434["text"] = "İşlem Türü"
GLabel_434.place(x=1110, y=460, width=400, height=40)

GLabel_199 = tk.Label(root)
ft = tkFont.Font(family='Times', size=11)
GLabel_199["font"] = ft
GLabel_199["fg"] = "#333333"
GLabel_199["justify"] = "center"
GLabel_199["text"] = "-"
GLabel_199["relief"] = "groove"
GLabel_199.place(x=980, y=510, width=662, height=40)

GLabel_909 = tk.Label(root)
ft = tkFont.Font(family='Times', size=18)
GLabel_909["font"] = ft
GLabel_909["fg"] = "#d02020"
GLabel_909["justify"] = "left"
GLabel_909["text"] = " "
GLabel_909.place(x=20, y=500, width=637, height=40)

GLabel_339 = tk.Label(root)
ft = tkFont.Font(family='Times', size=16)
GLabel_339["font"] = ft
GLabel_339["fg"] = "#333333"
GLabel_339["justify"] = "center"
GLabel_339["text"] = "Buzdolabı Toplam Ağırlık"
GLabel_339.place(x=730, y=350, width=220, height=50)

GLabel_406 = tk.Label(root)
ft = tkFont.Font(family='Times', size=11)
GLabel_406["font"] = ft
GLabel_406["fg"] = "#333333"
GLabel_406["justify"] = "center"
GLabel_406["text"] = "0"
GLabel_406["relief"] = "groove"
GLabel_406.place(x=740, y=400, width=200, height=40)

GListBox_701 = tk.Listbox(root)
GListBox_701["borderwidth"] = "1px"
ft = tkFont.Font(family='Times', size=10)
GListBox_701["font"] = ft
GListBox_701["fg"] = "#333333"
GListBox_701["justify"] = "center"
GListBox_701.place(x=980, y=80, width=329, height=368)


GLabel_571 = tk.Label(root)
ft = tkFont.Font(family='Times', size=16)
GLabel_571["font"] = ft
GLabel_571["fg"] = "#333333"
GLabel_571["justify"] = "center"
GLabel_571["text"] = "Buzdolabı İçindekiler"
GLabel_571.place(x=1040, y=40, width=200, height=25)

GListBox_882 = tk.Listbox(root)
GListBox_882["borderwidth"] = "1px"
ft = tkFont.Font(family='Times', size=10)
GListBox_882["font"] = ft
GListBox_882["fg"] = "#333333"
GListBox_882["justify"] = "center"
GListBox_882.place(x=1320, y=80, width=324, height=368)

GLabel_113 = tk.Label(root)
ft = tkFont.Font(family='Times', size=16)
GLabel_113["font"] = ft
GLabel_113["fg"] = "#333333"
GLabel_113["justify"] = "center"
GLabel_113["text"] = "Tüketilen Ürünler"
GLabel_113.place(x=1380, y=50, width=200, height=25)

GLabel_10 = tk.Label(root)
ft = tkFont.Font(family='Times', size=16)
GLabel_10["font"] = ft
GLabel_10["fg"] = "#333333"
GLabel_10["justify"] = "center"
GLabel_10["text"] = "Aylık Tüketilen Kalori"
GLabel_10.place(x=740, y=440, width=200, height=40)

GLabel_309 = tk.Label(root)
ft = tkFont.Font(family='Times', size=10)
GLabel_309["font"] = ft
GLabel_309["fg"] = "#333333"
GLabel_309["justify"] = "center"
GLabel_309["text"] = "0"
GLabel_309["relief"] = "groove"
GLabel_309.place(x=740, y=480, width=200, height=40)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    sonurun = ""
    oncekiurun = ""
    tespitsayisi = 0
    urunasama = 0
    sonkabuldeger = 0.0
    sonkabulurun = ""
    anlikdeger = 0.0
    sonurunagirlik = 0.0
    uruntespitet = 0
    islemturu = ""
    hassasolc=0
    agirlikolcmesayisi=0
    renk = (128, 128, 128)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                # s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                    s += names[int(c)]

                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=renk)
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)

            # LOGGER.info(f'{s}Done-a. ({t3 - t2:.3f}s)')


            #anlikdeger=float(veritabaniAgirlik())
            arduinoData = ser.readline().decode('utf-8').rstrip()
            anlikdeger = float(arduinoData)

            if (anlikdeger < 0):
                anlikdeger = 0

            if (hassasolc == 0):

                if (sonkabuldeger - anlikdeger > 50.0):
                    islemturu = "Ürün Çıkarılıyor"
                    hassasolc = 1
                    uruntespitet = 1

                if (sonkabuldeger - anlikdeger < -50):

                    hassasolc = 1

            else:
                agirlikolcmesayisi += 1
                if (agirlikolcmesayisi > 2):
                    if (anlikdeger > sonkabuldeger):
                        if(islemturu=="Ürün Eklenmesi için Ağırlık Bekleniyor"):

                            sonurunagirlik = anlikdeger - sonkabuldeger
                            sonkabuldeger = anlikdeger
                            UrunEkle(ClassNameToReal(sonkabulurun), sonurunagirlik)
                            sonkabulurun=""
                            islemturu = "Ürün Başarıyla Eklendi"
                            hassasolc = 0


                    else:

                        sonurunagirlik = sonkabuldeger - anlikdeger
                        sonkabuldeger = anlikdeger
                        islemturu = "Ürün Çıkartılması için Ürün Bekleniyor"
                        hassasolc = 0



            #print("Anlık:", anlikdeger)
            #print("Son Kabul Değer:", sonkabuldeger)
            #print("İşlem Türü:", islemturu)
            #print("Son Urun Agirlik:", sonurunagirlik)

            ArayuzUrunAgirliktxt(sonurunagirlik)
            ArayuzAnlikAgirliktxt(anlikdeger)
            ArayuzIslemTurutxt(islemturu)
            ArayuzBuzdolabiAgirlik(sonkabuldeger)


            # Stream results
            im0 = annotator.result()
            if view_img:
                ArayuzKamera(im0)
                # cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            sonurun = s


            if (sonurun == ""):
                ArayuzUruntxt("Ürün Bulunamadı")
                renk = (0, 0, 255)
                # img = cv2.imread("beyaz.png")
                # ArayuzUrunGorsel(img)
                # print("Boş")
                tespitsayisi = 0

                # Ürün Tespit Edilemedi
            elif (oncekiurun == sonurun):
                tespitsayisi += 1
                if (tespitsayisi < 2):
                    renk = (0, 255, 255)
                    ArayuzUruntxt("Doğrulanıyor")
                    print("Tespit Edilmeye Çalışılıyor " + sonurun + "   Tespit Sayısı " + str(tespitsayisi))
                else:
                    renk = (0, 255, 0)
                    ArayuzUrun(sonurun)

                    print("Tespit Edildi " + sonurun + "   Tespit Sayısı " + str(tespitsayisi))
                    if(islemturu=="Ürün Çıkartılması için Ürün Bekleniyor"):
                        sonkabulurun = sonurun
                        UrunCikar(ClassNameToReal(sonkabulurun),sonurunagirlik)
                        sonkabulurun=""
                        islemturu="Ürün Başarıyla Çıkartıldı"
                        tespitsayisi=0
                    else:
                        islemturu="Ürün Eklenmesi için Ağırlık Bekleniyor"
                        sonkabulurun=sonurun


            oncekiurun = sonurun

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def UrunCikar(Urun,Agirlik):


    # var mı kontrol et

    mydb = mysql.connector.connect(
        host="",
        user="",
        password="",
        database=""
    )

    mycursor = mydb.cursor()
    sql = "SELECT * FROM icindekiler Where Urun='" + Urun + "'"
    mycursor.execute(sql)

    myresult = mycursor.fetchall()

    degervar = 0
    oncekiid = 0
    oncekiagirlik = 0
    for x in myresult:
        degervar = 1
        print("degervar")
        print(x[1])
        oncekiid = x[0]
        oncekiagirlik = x[2]

    sonagirlik = float(oncekiagirlik) - float(Agirlik)
    if(sonagirlik<=0):
        Agirlik = float(oncekiagirlik)
    if(sonagirlik>0):
        if (degervar == 1):
            mydb = mysql.connector.connect(
                host="",
                user="",
                password="",
                database=""
            )

            mycursor = mydb.cursor()


            sql = "UPDATE icindekiler SET agirlik ='" + str(sonagirlik) + "' WHERE id = '" + str(oncekiid) + "'"
            mycursor.execute(sql)
            mydb.commit()
            print(mycursor.rowcount, "record inserted.")
    else:

        mydb = mysql.connector.connect(
            host="",
            user="",
            password="",
            database=""
        )

        mycursor = mydb.cursor()

        sql = "DELETE FROM icindekiler WHERE id = '"+str(oncekiid)+"'"

        mycursor.execute(sql)

        mydb.commit()

        print(mycursor.rowcount, "record(s) deleted")

    #tuketilenlere ekle
    mydb = mysql.connector.connect(
        host="",
        user="",
        password="",
        database=""
    )
    now = datetime.now()
    formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
    kalori=kaloriHesabi(Urun,Agirlik)
    mycursor = mydb.cursor()
    sql = "INSERT INTO tuketilenler (urun, agirlik,kalori,tarih) VALUES (%s, %s,%s,%s)"
    val = (Urun, Agirlik,kalori,formatted_date)
    mycursor.execute(sql, val)
    mydb.commit()
    print(mycursor.rowcount, "record inserted.")

    a = "Ürün Çıkartıldı " + str(Urun) + "-" + str(Agirlik) + " Gram"
    ArayuzConsole(a)

    UrunListesiGuncelle()
    TuketilenlerListesiGuncelle()
    aylikKaloriGuncelle()
def UrunEkle(Urun,Agirlik):


    #var mı kontrol et

    mydb = mysql.connector.connect(
        host="",
        user="",
        password="",
        database=""
    )

    mycursor = mydb.cursor()
    sql = "SELECT * FROM icindekiler Where Urun='"+Urun+"'"
    mycursor.execute(sql)

    myresult = mycursor.fetchall()

    degervar=0
    oncekiid=0
    oncekiagirlik=0
    for x in myresult:
        degervar = 1
        print("degervar")
        print(x[1])
        oncekiid=x[0]
        oncekiagirlik=x[2]



    #eğer yoksa ekle
    if(degervar==0):
        print("Değer Yok Ekleniyor")
        mydb = mysql.connector.connect(
            host="",
            user="",
            password="",
            database=""
        )
        mycursor = mydb.cursor()
        sql = "INSERT INTO icindekiler (urun, agirlik) VALUES (%s, %s)"
        val = (Urun, Agirlik)
        mycursor.execute(sql, val)
        mydb.commit()
        print(mycursor.rowcount, "record inserted.")

    #varsa Güncelle
    if(degervar==1):
        mydb = mysql.connector.connect(
            host="",
            user="",
            password="",
            database=""
        )

        mycursor = mydb.cursor()
        sonagirlik=float(oncekiagirlik)+float(Agirlik)

        sql = "UPDATE icindekiler SET agirlik ='"+str(sonagirlik)+"' WHERE id = '"+str(oncekiid)+"'"
        mycursor.execute(sql)
        mydb.commit()
        print(mycursor.rowcount, "record inserted.")

    a = "Ürün Eklendi " + str(Urun) + "-" + str(Agirlik) + " Gram"
    ArayuzConsole(a)

    UrunListesiGuncelle()
    TuketilenlerListesiGuncelle()
    aylikKaloriGuncelle()


def kaloriHesabi(urun, eksilen_agirlik):
    kcal=0
    if urun==  "icim Sut":
        kcal=(42/100)*eksilen_agirlik
    elif urun==  "Calve Cheddarsos":
        kcal=(370/100)*eksilen_agirlik
    elif urun==  "Pınar Suzme Peynir":
        kcal=(207/100)*eksilen_agirlik
    elif urun==  "Calve Ranchsos":
        kcal=(350/100)*eksilen_agirlik
    elif urun==  "Tat Mayonez":
        kcal=(627/100)*eksilen_agirlik
    elif urun==  "Superfresh Misir":
        kcal=(111/100)*eksilen_agirlik
    elif urun==  "Sek Yogurt":
        kcal=(80/100)*eksilen_agirlik
    elif urun==  "Binvezir Peynir":
        kcal=(289/100)*eksilen_agirlik
    elif urun==  "Elma":
        kcal=(52/100)*eksilen_agirlik
    elif urun ==  "Portakal":
        kcal = (44/ 100) * eksilen_agirlik
    elif urun ==  "Limon":
        kcal = (28/ 100) * eksilen_agirlik
    elif urun ==  "Zeytin":
        kcal = (144/ 100) * eksilen_agirlik
    elif urun == "Biber":
        kcal = (20/ 100) * eksilen_agirlik
    kcal=round(kcal, 2)
    return kcal;


def UrunListesiGuncelle():

    GListBox_701.delete(0, 200)

    mydb = mysql.connector.connect(
        host="",
        user="",
        password="",
        database=""
    )

    mycursor = mydb.cursor()
    sql = "SELECT * FROM icindekiler Where 1"
    mycursor.execute(sql)

    myresult = mycursor.fetchall()
    for x in myresult:

        oncekiid = x[0]
        gelenurun=x[1]
        oncekiagirlik = x[2]
        GListBox_701.insert(oncekiid,gelenurun+" "+oncekiagirlik)

def TuketilenlerListesiGuncelle():

    GListBox_882.delete(0, 100)

    mydb = mysql.connector.connect(
        host="",
        user="",
        password="",
        database=""
    )

    mycursor = mydb.cursor()
    sql = "SELECT * FROM tuketilenler Where 1"
    mycursor.execute(sql)

    myresult = mycursor.fetchall()
    for x in myresult:

        oncekiid = x[0]
        gelenurun=x[1]
        oncekiagirlik = x[2]
        oncekikalori = x[3]
        oncekitarih = str(x[4])
        GListBox_882.insert(oncekiid,gelenurun+" "+oncekiagirlik+" "+oncekikalori+" "+oncekitarih)

def aylikKaloriGuncelle():
    mydb = mysql.connector.connect(
        host="",
        user="",
        password="",
        database=""
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT sum(kalori) FROM tuketilenler WHERE MONTH(tarih)=MONTH(NOW()) AND YEAR(tarih)=YEAR(NOW())")

    myresult = mycursor.fetchall()
    degervar=0
    for x in myresult:
        degervar=1
        GLabel_309['text'] = x[0]

    if(degervar==0):
        GLabel_309['text']=0

def veritabaniAgirlik():
    mydb = mysql.connector.connect(
        host="",
        user="",
        password="",
        database=""
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM agirlik")

    myresult = mycursor.fetchall()

    for x in myresult:
        #print(x[1])
        return(x[1])
def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best3.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '1', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def ArayuzKamera(img):
    # img=cv2.imread("gorsel.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(img))
    GLabel_854['image'] = img
    root.update()


def ArayuzUrun(classname):
    Name = ClassNameToReal(classname)
    ArayuzUruntxt(Name)
    # img = cv2.imread(classname+".png")
    # ArayuzUrunGorsel(img)


def ArayuzUruntxt(Name):
    GLabel_224['text'] = Name


def ArayuzUrunAgirliktxt(Name):
    GLabel_282['text'] = Name


def ArayuzAnlikAgirliktxt(Name):
    GLabel_929['text'] = Name


def ArayuzIslemTurutxt(Name):
    GLabel_199['text'] = Name

def ArayuzConsole(Name):
    GLabel_909['text'] = Name

def ArayuzBuzdolabiAgirlik(Name):
    GLabel_406['text'] = Name

def ArayuzUrunGorsel(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = ImageTk.PhotoImage(Image.fromarray(img))
    image = img

    # GLabel_299['image'] = image
    root.update()


def ClassNameToReal(classname):
    realName = "";
    if (classname == "icim_sut"):
        realName = "icim Sut"
    if (classname == "calve_cheddarsos"):
        realName = "Calve Cheddarsos"
    if (classname == "elma"):
        realName = "Elma"
    if (classname == "limon"):
        realName = "Limon"
    if (classname == "pinar_suzmepeynir"):
        realName = "Pınar Suzme Peynir"
    if (classname == "portakal"):
        realName = "Portakal"
    if (classname == "zeytin"):
        realName = "Zeytin"
    if (classname == "biber"):
        realName = "Biber"
    if (classname == "calve_ranchsos"):
        realName = "Calve Ranchsos"
    if (classname == "tat_mayonez"):
        realName = "Tat Mayonez"
    if (classname == "superfresh_misir"):
        realName = "Superfresh Misir"
    if (classname == "sek_yogurt"):
        realName = "Sek Yogurt"
    if (classname == "binvezir_peynir"):
        realName = "Binvezir Peynir"
    return realName


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)




