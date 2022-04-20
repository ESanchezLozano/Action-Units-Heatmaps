
import cv2, time, dlib, matplotlib.pyplot as plt, numpy as np
import AUmaps

AUdetector = AUmaps.AUdetector('shape_predictor_68_face_landmarks.dat',enable_cuda=True)

cam = cv2.VideoCapture(1)


fig = plt.figure(figsize=plt.figaspect(.5))
axs = fig.subplots(5, 2)

# Init subplots and images within
implots = []
for ax in axs.reshape(-1):
    ax.axis('off')
    implots.append(ax.imshow(np.zeros((256, 256))))

tstart_time = time.time()
nframes = 0
while True:
	start_time = time.time()
	_, img = cam.read()
	try:
		# Downscale webcam image 2x to speed things up
		img = cv2.resize(img, None, fx = 0.5, fy = 0.5)

		# Optionally flip webcam image, probably not relevant
		# img = cv2.flip(img, 1)

		# Optionally display webcam image with opencv
		# cv2.imshow('Action Unit Heatmaps - Press Q to exit!', img)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	break

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		nframes += 1

		pred,map,img = AUdetector.detectAU(img)
		for j in range(0,5):
			resized_map = dlib.resize_image(map[j,:,:].cpu().data.numpy(),rows=256,cols=256)

		# Update face image subplot
			implots[2*j].set_data(img)

		# Update heatmap subplot
			implots[2*j+1].set_data(resized_map)

			# Set correct heatmap limits
			resized_map_flat = resized_map.flatten()
			implots[2*j+1].set_clim(min(resized_map_flat), max(resized_map_flat))

		# To plot heatmaps the original way - looks identical to the new way!
		# ax = fig.add_subplot(5,2,2*j+1)
		# ax.imshow(resized_map)
		# ax.axis('off')

		plt.pause(0.001)
		plt.draw()

		elapsed_time = time.time() - start_time
		print(' ** FPS Elapsed: {0:.3f}'.format(1.0 / elapsed_time))
	except Exception:
	    pass

# Close camera
cam.release()

# If webcam images shown with opencv, close window
# cv2.destroyAllWindows()

telapsed_time = time.time() - tstart_time
print('\n ** Mean FPS Elapsed: {0:.3f} \n'.format(1.0 / (telapsed_time / nframes)))
