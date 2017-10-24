from mobilescanner_final_curr import normalizeImg, displayImg

#x,y = normalizeImg('sample1.jpg'); displayImg(x, True); displayImg(y, True)
x,y = normalizeImg('testempty.jpg'); displayImg(x, True); displayImg(y, True)
#x,y = normalizeImg('sample4.jpg'); displayImg(x, True); displayImg(y, True)  #full contour
#x,y = normalizeImg('sample5.jpg'); displayImg(x, True); displayImg(y, True)  #full contour
#x,y = normalizeImg('.\\newimgs\\form2\\fullempty.jpg'); displayImg(x, True); displayImg(y, True)
#x,y = normalizeImg('.\\newimgs\\form2\\fullfilled.jpg'); displayImg(x, True); displayImg(y, True)
x,y = normalizeImg('.\\newimgs\\form2\\testempty.jpg'); displayImg(x, True); displayImg(y, True)
#x,y = normalizeImg('.\\newimgs\\form2\\testfilled.jpg'); displayImg(x, True); displayImg(y, True)

#x,y = normalizeImg('.\\newimgs\\form1\\fullempty.jpg'); displayImg(x, True); displayImg(y, True)
#x,y = normalizeImg('.\\newimgs\\form1\\fullfilled.jpg'); displayImg(x, True); displayImg(y, True)
#normalizeImg('form.jpg')

#normalizeImg('.\\newimgs\\form2\\fullempty.jpg')

if False:
	aa = normalizeImg('.\\newimgs\\form2\\stitched_.png', stitched=True)
	displayImg(aa[0], True)
	displayImg(aa[1], True)
	#pdb.set_trace