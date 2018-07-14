import numpy as np
import math
import cv2
import moviepy.editor as mpe

#Convert a number to a color based on its position between a minimum and maximum value in a heatmap style
def convertRGB(minimum, maximum, value):
	minimum, maximum = float(minimum), float(maximum)    
	halfmax = (minimum + maximum) / 2
	if minimum <= value <= halfmax:
		r = 0
		g = int( 255./(halfmax - minimum) * (value - minimum))/5
		b = int( 255. + -255./(halfmax - minimum)  * (value - minimum))
		return (r,g,b)    
	elif halfmax < value <= maximum:
		r = int( 255./(maximum - halfmax) * (value - halfmax))
		g = int( 255. + -255./(maximum - halfmax)  * (value - halfmax))/5
		b = 0
		return (r,g,b)
		

#Arguments
#eyeCoordinates: A list of lists of integers with the format = [[X,Y],[X,Y]]
#eegData: A list of values between -1 and 1 
#focusRadius: Integer
#auraRadius: Integer
#fps: Integer
#openVidPath: File Path
#openAudioPath: File Path
#saveVidPath: File Path
#headless: Boolean

#Create a new video using EEG data and eye gaze coordinates to show a viewer's attention while watching a video
def focusEditor(eyeCoordinates,eegData,focusRadius = 40,auraRadius = 50,fps,openVidPath,openAudioPath,saveVidPath,headless = False):
	
	# Determine dimensions of the video, define the codec, and create VideoWriter/VideoCapture objects
	codec = cv2.VideoWriter_fourcc(*'mpeg')
	vidCapture = cv2.VideoCapture(openVidPath)
	vidSize = (int(vidCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vidCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	vidWriter = cv2.VideoWriter(saveVidPath,codec, fps, vidSize, True)
	
	#Convert EEG Data to color values
	auraColors = [convertRGB(-1, 1, x) for x in eegData]
	
	#Setup arrays for pixel positions along X and Y axis
	X = [x for x in range(0,vidSize[0])]
	Y = [y for y in range(0,vidSize[1])]	
	
	#Determine amount of EEG and Eye Tracker data, and begin a count of the number of frames captured
	eegDataLength = len(eegData)
	eyeCoordinatesLength = len(eyeCoordinates)	
	frameCount = 0
	
	#Open the VideoCapture object and begin reading frames
	while(vidCapture.isOpened()):
		ret, frame = vidCapture.read()
		
		#Begin editing process for a frame only if a frame is present and there is EEG and Eye Tracker data available
		if ret == True and frameCount < eegDataLength and frameCount < eyeCoordinatesLength:
			
			#Split the frame into red, green, and blue channels
			b,g,r = cv2.split(frame)
			
			#Determine the distances from the eye tracker coordinates for every position on the X-axis and Y-axis
			xyCoord = eyeCoordinates[frameCount]
			distanceX = [abs(xyCoord[0]-X[x])**2 for x in X]
			distanceY = [abs(xyCoord[1]-Y[y])**2 for y in Y]

			#Setup matrices that will define the dimensions of the aura circle, focus circle, and background
			distancesMatrix = [[math.sqrt(x + y) for x in distanceX] for y in distanceY]
			auraSizeMatrix = [[0 if x <= auraRadius else 1 for x in y] for y in distancesMatrix]
			focusSizeMatrix = [[1 if x <= focusRadius else 0 for x in y] for y in distancesMatrix]
			negFocusSizeMatrix = [[0 if x <= focusRadius else 1 for x in y] for y in distancesMatrix]
			
			#Create both a list and counter to hold the edited channels and iterate over them, respectively
			processedChannels = []
			channelCount = 0
			
			#For each channel, combine matrices with frame's color channel to form a center of focus, aura surrounding the focus, and fuzzy background
			for channel in [r,g,b]:
				aura = [[auraColors[frameCount][channelCount] if x == 0 else 0 for x in y] for y in auraSizeMatrix]		
				background = auraSizeMatrix * channel
				combinedMatrix = background + aura
				fuzzyBackground = cv2.filter2D(combinedMatrix,-1,np.ones((20,20),np.float32)/400)
				focus = focusSizeMatrix * channel
				finalMatrix = focus + (fuzzyBackground * negFocusSizeMatrix) 
				processedChannels.append(finalMatrix)
				channelCount += 1
			
			#Combine all three channels again and write the newly created frame to file
			finalRGB = np.dstack((processedChannels[2],processedChannels[1],processedChannels[0]))
			vidWriter.write(np.uint8(finalRGB))
			frameCount += 1
			
			#Decide whether or not to show the video as it is being processed.
			if headless == False:
				cv2.imshow('frame',np.uint8(finalRGB))
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
		
		else:
			break
	
	#End all processes we have been using
	vidCapture.release()
	vidWriter.release()
	cv2.destroyAllWindows()
	
	#Save the edited video clip with its corresponding audio clip edited back in
	videoclip = mpe.VideoFileClip(saveVidPath)
	videoclip.audio = mpe.AudioFileClip(openAudioPath)
	videoclip.write_videofile(saveVidPath)