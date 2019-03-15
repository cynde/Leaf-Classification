#by15055-15074-15102

#IMPORT
from libFP import *
from tkinter import *
from PIL import ImageTk
from tkinter import filedialog
from PIL import Image as Im
#IMPORT

#global
file_path=None
fol=None
#global

def proc(pred):
	global file_path,fol
	#path_file=file_path.split("/")
	#print "doSomething"
#	img = cv2.imread(file_path)
	vector = toVector(file_path)
	res = doPredict(vector)
	res = res.tolist()
	label = ["Ulmus carpinifolia", "Acer", "Salix aurita", "Quercus", "Alnus incana", "Betula pubescens", "Salix alba 'sericea'", "Populus tremula", "Ulmus glabra", "Sorbus aucuparia", "Salix sinerea", "Populus", "Tilia", "Sorbus intermedia", "Fagus silvatica"]
	for i in range (0,15):
		if res[0][i] == 1:
			pred['text']= "Prediction: " + label[i] + " (" + str(i+1) + ")"
			break
#	pred['text']=res	

def choose_file(event,home,img_open,img,panel_image):
	global file_path
	choose_img=filedialog.askopenfilename(initialdir="dataset",title="Select File",filetypes=(("all files","*"),("jpeg files","*.jpg")))
	file_path=choose_img
	
	img_open=Im.open(file_path)
	img_open=img_open.resize((250, 400), Im.ANTIALIAS)
	img=ImageTk.PhotoImage(img_open)
	panel_image.configure(image=img)
	panel_image.image=img
	
def getText(ent):
	global fol
	fol=ent.get()
	

def main():
	global fol
	home=Tk()
	home.geometry("600x650")
	home.title("Image Label Predict")
	
	frame_top=Frame(home)
	frame_top.pack()
	
	frame_bot=Frame(home)
	frame_bot.pack()
	
	img_open=Im.open("resources/choose_file.jpg")
	img_open=img_open.resize((150, 150), Im.ANTIALIAS)
	img=ImageTk.PhotoImage(file="resources/choose_file.jpg")
	panel_image=Label(frame_top,image=img)
	panel_image.image=img
	panel_image.pack()
	
	prediction=Label(frame_top,text="Prediction")
	prediction.config(font=("Segoe UI", 17))
	
	proc_button=Button(frame_top,text="Process !",command=lambda:proc(prediction), bg = '#edcb62', bd=0, activebackground='#d5b658')
	proc_button.config(font=("Segoe UI", 12))
	proc_button.pack(ipadx = 20, ipady = 5, pady = 10)
	
	prediction.pack(pady = 10)
	panel_image.bind("<Button-1>",lambda event,a=frame_top,b=img_open,c=img,d=panel_image:choose_file(event,a,b,c,d))
	
	#dataset_panel=Toplevel()
	#dataset_panel.geometry("200x200")
	#dataset_panel.title("Input Dataset Location")
	
	
	#Label(frame_bot).grid(row=0)
	#Label(frame_bot).grid(row=1)
	#Label(frame_bot,text="Set directory name here").grid(row=2)
	#entry=Entry(frame_bot)
	#entry.grid(row=3,column=0)
	
	#Button(frame_bot,text="Set",command=lambda:getText(entry)).grid(row=3,column=1)
	
	home.mainloop()
	

if __name__=="__main__":
	main()
